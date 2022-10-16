from enum import Enum
import functools
import numpy as np
from typing import Callable

import torch
import torch.nn as nn
import torch.distributions as td

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.taac_loss import TAACTDLoss
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import LossInfo, namedtuple, TimeStep
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, dist_utils, losses, math_ops, tensor_utils
from alf.utils.conditional_ops import conditional_update
from alf.utils.summary_utils import safe_mean_hist_summary

TaacState = namedtuple("TaacState", ["action", "repeats"], default_value=())

TaacActorInfo = namedtuple(
    "TaacActorInfo", ["actor_loss", "b1_a_entropy", "beta_entropy"], default_value=())

TaacCriticInfo = namedtuple(
    "TaacCriticInfo", ["critics", "target_critic"], default_value=())

TaacInfo = namedtuple(
    "TaacInfo", [
        "reward", "step_type", "action", "prev_action", "discount",
        "action_distribution", "rollout_b", "b", "actor", "critic", "alpha",
        "repeats"], default_value=())

TaacLossInfo = namedtuple('TaacLossInfo', ('actor', 'critic', 'alpha'))

Distributions = namedtuple("Distributions", ["beta_dist", "b1_action_dist"])  # β=Q^/Q-+Q^, π(a^|)

ActPredOutput = namedtuple("ActPredOutput", ["dists", "b", "actions", "q_values2"], default_value=())

Mode = Enum('AlgorithmMode', ('predict', 'rollout', 'train'))

@alf.configurable
class TaacAlgorithm(OffPolicyAlgorithm):
    r"""Temporally abstract actor-critic algorithm.

    In a nutsell, for inference TAAC adds a second stage that chooses between a
    candidate trajectory :math:`\hat{\action}` output by an SAC actor and the previous
    trajectory :math:`\action^-`.

    - For policy evaluation, TAAC uses a compare-through Q operator for TD backup
    by re-using state-action sequences that have shared actions between rollout and training.
    - For policy improvement, the new actor gradient is approximated by multiplying a scaling factor
    to the :math:`\frac{\partial Q}{\partial a}` dQ/da term in the original SAC’s actor
    gradient, where the scaling factor is the optimal probability of choosing
    the :math:`\hat{\action}` in the second stage.

    Different sub-algorithms implement different forms of the 'trajectory' concept,
    for example, it can be a constant function representing the same action, or
    a quadratic function.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 reward_weights=None,
                 num_critic_replicas=2,
                 epsilon_greedy=None,
                 env=None,
                 config: TrainerConfig = None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 critic_loss_ctor=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 debug_summaries=False,
                 randomize_first_state_action=False,
                 b1_advantage_clipping=None,
                 max_repeat_steps=None,
                 target_entropy=None,
                 dynamic_low=-1,
                 dynamic_high=4,
                 name="TaacAlgorithm"):
        r"""
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (BoundedTensorSpec): representing the continuous action.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network will be called to sample continuous
                actions.
            critic_network_cls (Callable): is used to construct critic network.
                for estimating ``Q(s,a)`` given that the action is continuous.
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the q values is used.
            num_critic_replicas (int): number of critics to be used. Default is 2.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``alf.get_config_value(TrainerConfig.epsilon_greedy)``
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``TAACTDLoss`` will be used.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            debug_summaries (bool): True if debug summaries should be created.
            randomize_first_state_action (bool): whether to randomize ``state.action``
                at the beginning of an episode during rollout and training.
                Potentially this helps exploration. This was turned off in
                Yu et al. 2021.
            b1_advantage_clipping (None|tuple[float]): option for clipping the
                advantage (defined as :math:`Q(s,\hat{\action}) - Q(s,\action^-)`) when
                computing :math:`\beta_1`. If not ``None``, it should be a pair
                of numbers ``[min_adv, max_adv]``.
            max_repeat_steps (None|int): the max number of steps to repeat during
                rollout and evaluation. This value doesn't impact the switch
                during training.
            target_entropy (Callable|tuple[Callable]|None): If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. If ``None``, a default entropy will
                be calculated. To set separate entropy targets for the two
                stage policies, this argument can be a tuple of two callables.
            name (str): name of the algorithm
        """
        assert len(nest.flatten(action_spec)) == 1 and action_spec.is_continuous, (
            "Only support a single continuous action!")

        self._action_spec = action_spec
        self._num_critic_replicas = num_critic_replicas
        critic_networks, actor_network = self._make_networks_impl(
            observation_spec, action_spec, actor_network_cls, critic_network_cls)

        train_state_spec = TaacState(
            action=self._action_spec,
            repeats=TensorSpec(shape=(), dtype=torch.int64))

        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            reward_weights=reward_weights,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        log_alpha = (nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(())))
        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, list(log_alpha))

        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value('TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy
        self._log_alpha = log_alpha
        self._log_alpha_paralist = nn.ParameterList(list(log_alpha))
        self._actor_network = actor_network
        self._critic_networks = critic_networks
        self._target_critic_networks = self._critic_networks.copy(name='target_critic_networks')
        self.register_buffer("_training_started", torch.zeros((), dtype=torch.bool))

        self.dynamic_low = dynamic_low
        self.dynamic_high = dynamic_high

        # % cirtic
        if critic_loss_ctor is None:
            critic_loss_ctor = TAACTDLoss
        critic_loss_ctor = functools.partial(critic_loss_ctor, debug_summaries=debug_summaries)

        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(critic_loss_ctor(name="critic_loss%d" % (i + 1)))
        self._gamma = self._critic_losses[0]._gamma

        self._b_spec = BoundedTensorSpec(shape=(), dtype='int64', maximum=1)
        if not isinstance(target_entropy, tuple):
            target_entropy = (target_entropy,) * 2
        self._target_entropy = nest.map_structure(
            lambda t, spec: _set_target_entropy(self.name, t, [spec]),
            target_entropy, (self._b_spec, action_spec))

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

    def _make_networks_impl(self, observation_spec, action_spec,
                            actor_network_cls, critic_network_cls):
        action_embedding = torch.nn.Sequential(alf.layers.FC(input_size=action_spec.numel,
                                                             output_size=observation_spec.numel))
        actor_network = actor_network_cls(
            input_tensor_spec=(observation_spec, action_spec),
            input_preprocessors=(alf.layers.Detach(), action_embedding),
            preprocessing_combiner=nest_utils.NestConcat(),
            action_spec=action_spec)
        critic_network = critic_network_cls(
            input_tensor_spec=(observation_spec, action_spec),
            action_preprocessing_combiner=nest_utils.NestConcat())
        critic_networks = critic_network.make_parallel(self._num_critic_replicas)

        return critic_networks, actor_network

    def _build_beta_dist(self, q_values2):
        with torch.no_grad():
            beta_alpha = self._log_alpha[0].exp().detach()
            q_values2 = q_values2 / torch.clamp(beta_alpha, min=1e-10)
            q_values2 = q_values2 - torch.max(q_values2, dim=-1, keepdim=True)[0]
            beta_dist = td.Categorical(logits=q_values2)
        return beta_dist

    def _compute_beta_and_action(self, observation, state, epsilon_greedy, mode):
        # % 1. sac_base: a^ ~ π(a^|s,a-)
        b1_action_dist, _ = self._actor_network((observation, state.action))
        if mode == Mode.predict:
            b1_action = dist_utils.epsilon_greedy_sample(b1_action_dist, epsilon_greedy)
        else:
            b1_action = dist_utils.rsample_action_distribution(b1_action_dist)

        # % 2. taac
        """Update the current trajectory ``action`` by moving one step ahead."""
        """Compute a new trajectory ``action`` given a new action."""
        b0_action = state.action  # a-

        with torch.no_grad():
            q_0 = self._compute_critics(self._critic_networks, observation, b0_action)
        q_1 = self._compute_critics(self._critic_networks, observation, b1_action)

        q_values2 = torch.stack([q_0, q_1], dim=-1)
        beta_dist = self._build_beta_dist(q_values2)

        # % 3. a^
        if mode == Mode.predict:
            b = dist_utils.epsilon_greedy_sample(beta_dist, epsilon_greedy)
        else:
            b = dist_utils.sample_action_distribution(beta_dist)

        dists = Distributions(beta_dist=beta_dist, b1_action_dist=b1_action_dist)
        return ActPredOutput(
            dists=dists,
            b=b,
            actions=(b0_action, b1_action),
            q_values2=q_values2)

    def predict_step(self, inputs: TimeStep, state):
        ap_out, new_state = self._predict_action(inputs.observation, state,
                                                 epsilon_greedy=self._epsilon_greedy, mode=Mode.predict)
        info = TaacInfo(action_distribution=ap_out.dists, b=ap_out.b)
        return AlgStep(output=new_state.action, state=new_state, info=info)

    def rollout_step(self, inputs: TimeStep, state):
        ap_out, new_state = self._predict_action(inputs.observation, state, mode=Mode.rollout)
        info = TaacInfo(
            action_distribution=ap_out.dists,
            prev_action=state.action,  # a-, for getting randomized action in training
            action=new_state.action,  # a, real action for critic training
            b=ap_out.b,  # b, for Info
            repeats=state.repeats)
        return AlgStep(output=new_state.action, state=new_state, info=info)

    def _predict_action(self, observation, state, epsilon_greedy=None, mode=Mode.rollout):
        """selectively update with new actions"""
        ap_out = self._compute_beta_and_action(observation, state, epsilon_greedy, mode)

        if not common.is_eval() and not self._training_started:
            b = self._b_spec.sample(observation.shape[:1])
            b1_action = self._action_spec.sample(observation.shape[:1])
            ap_out = ap_out._replace(b=b, actions=(ap_out.actions[0], b1_action))

        def _b1_action(b1_action, state):
            new_state = state._replace(action=b1_action, repeats=torch.zeros_like(state.repeats))
            return new_state

        b0_action, b1_action = ap_out.actions  # a-, a^
        condition = ap_out.b.to(torch.bool)

        new_state = conditional_update(
            target=state,
            cond=condition,
            func=_b1_action,
            b1_action=b1_action,
            state=state)

        new_state = new_state._replace(repeats=new_state.repeats + 1)
        return ap_out, new_state

    def _compute_critics(self, critic_net, observation, action,
                         replica_min=True):
        critics, _ = critic_net((observation, action))
        if replica_min:
            critics = critics.min(dim=1)[0]

        return critics

    # % train_step()
    def train_step(self, inputs: TimeStep, state: TaacState, rollout_info: TaacInfo):
        self._training_started.fill_(True)
        ap_out, new_state = self._predict_action(inputs.observation, state=state, mode=Mode.train)

        beta_dist = ap_out.dists.beta_dist
        b1_action_dist = ap_out.dists.b1_action_dist
        b0_action, b1_action = ap_out.actions
        q_values2 = ap_out.q_values2

        b1_a_entropy = -dist_utils.compute_log_probability(b1_action_dist, b1_action)
        beta_entropy = beta_dist.entropy()

        actor_loss = self._actor_train_step(
            b1_action, b1_a_entropy, beta_dist, beta_entropy, q_values2)
        critic_info = self._critic_train_step(
            inputs, rollout_info.action, b0_action, b1_action, beta_dist)
        alpha_loss = self._alpha_train_step(beta_entropy, b1_a_entropy)

        action = new_state.action
        """compare beta action(ap_out.b) :math:`b~_n` sampled from the current policy 
           with the historical rollout beta action :math:`b_n` step by step"""
        info = TaacInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            rollout_b=rollout_info.b,
            action_distribution=ap_out.dists,
            actor=actor_loss,
            critic=critic_info,
            b=ap_out.b,
            alpha=alpha_loss,
            repeats=state.repeats)
        return AlgStep(output=action, state=new_state, info=info)

    def _actor_train_step(self, a, b1_a_entropy, beta_dist, beta_entropy, q_values2):
        alpha = self._log_alpha[1].exp().detach()
        q_a = beta_dist.probs[:, 1].detach() * q_values2[:, 1]

        def actor_loss_fn(dqda, action):
            loss = 0.5 * losses.element_wise_squared_loss((dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        dqda = nest_utils.grad(a, q_a.sum())  # dq/da
        actor_loss = nest.map_structure(actor_loss_fn, dqda, a)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))
        actor_loss -= alpha * b1_a_entropy  # dq/da - πlogπ

        actor_info = LossInfo(
            loss=actor_loss,
            extra=TaacActorInfo(actor_loss=actor_loss, b1_a_entropy=b1_a_entropy, beta_entropy=beta_entropy))
        return actor_info

    def _critic_train_step(self, inputs: TimeStep, rollout_action, b0_action, b1_action, beta_dist):
        """compute target_q"""
        with torch.no_grad():
            target_q_0 = self._compute_critics(
                self._target_critic_networks,
                inputs.observation,
                b0_action)
            target_q_1 = self._compute_critics(
                self._target_critic_networks,
                inputs.observation,
                b1_action)

            beta_probs = beta_dist.probs
            target_critic = (beta_probs[..., 0] * target_q_0 + beta_probs[..., 1] * target_q_1)

        critics = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_action,
            replica_min=False)
        return TaacCriticInfo(critics=critics, target_critic=target_critic)

    def _alpha_train_step(self, beta_entropy, action_entropy):
        """α * -πlogπ"""
        alpha_loss = (self._log_alpha[1] * (action_entropy - self._target_entropy[1]).detach())
        alpha_loss += (self._log_alpha[0] * (beta_entropy - self._target_entropy[0]).detach())
        return alpha_loss

    def calc_loss(self, info: TaacInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_loss = info.actor

        with alf.summary.scope(self._name):
            alf.summary.scalar("train_repeats/mean", torch.mean(info.repeats.to(torch.float32)))

        loss = actor_loss.loss + alpha_loss + critic_loss.loss
        extra = TaacLossInfo(actor=actor_loss.extra, critic=critic_loss.extra, alpha=alpha_loss)
        return LossInfo(loss=loss, extra=extra)

    def _calc_critic_loss(self, info: TaacInfo):
        """r + (Q'-logπ) - Q"""
        critic_info = info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            kwargs = dict(
                info=info,
                value=critic_info.critics[:, :, i, ...],
                target_value=critic_info.target_critic)
            critic_losses.append(l(**kwargs).loss)

        critic_loss = math_ops.add_n(critic_losses)
        return LossInfo(
            loss=critic_loss,
            extra=critic_loss / float(self._num_critic_replicas))

    def after_update(self, root_inputs, info: TaacInfo):
        self._update_target()

    def after_train_iter(self, root_inputs, rollout_info=None):
        # print(root_inputs.reward[0][0])
        pass

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']

    def summarize_rollout(self, experience):
        repeats = experience.rollout_info.repeats.reshape(-1)
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("rollout_repeats/mean", torch.mean(repeats.to(torch.float32)))
