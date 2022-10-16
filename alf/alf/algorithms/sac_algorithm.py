"""Soft Actor Critic Algorithm."""

from absl import logging
import numpy as np
import functools
from enum import Enum

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import QNetwork, QRNNNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops

ActionType = Enum('ActionType', ('Discrete', 'Continuous', 'Mixed'))
SacActionState = namedtuple("SacActionState", ["actor_network", "critic"], default_value=())
SacCriticState = namedtuple("SacCriticState", ["critics", "target_critics"])

SacState = namedtuple("SacState", ["action", "actor", "critic"], default_value=())

SacCriticInfo = namedtuple("SacCriticInfo", ["critics", "target_critic"])
SacActorInfo = namedtuple("SacActorInfo", ["actor_loss", "neg_entropy"], default_value=())
SacInfo = namedtuple(
    "SacInfo", [
        "reward", "step_type", "discount", "action", "action_distribution",
        "actor", "critic", "alpha", "log_pi"
    ], default_value=())

SacLossInfo = namedtuple('SacLossInfo', ["actor", "critic", "alpha"])

def _set_target_entropy(name, target_entropy, flat_action_spec):
    """A helper function for computing the target entropy under different
    scenarios of ``target_entropy``.

    Args:
        name (str): the name of the algorithm that calls this function.
        target_entropy (float|Callable|None): If a floating value, it will return as it is.
            If a callable function, then it will be called on the action spec to
            calculate a target entropy. If ``None``, a default entropy will be calculated.
        flat_action_spec (list[TensorSpec]): a flattened list of action specs.
    """
    if target_entropy is None or callable(target_entropy):
        if target_entropy is None:
            target_entropy = dist_utils.calc_default_target_entropy
        target_entropy = np.sum(list(map(target_entropy, flat_action_spec)))
        logging.info("Target entropy is calculated for {}: {}.".format(name, target_entropy))
    else:
        logging.info("User-supplied target entropy for {}: {}".format(name, target_entropy))
    return target_entropy

@alf.configurable
class SacAlgorithm(OffPolicyAlgorithm):
    r"""Soft Actor Critic algorithm, described in:

    ::

        Haarnoja et al "Soft Actor-Critic Algorithms and Applications", arXiv:1812.05905v2

    There are 3 points different with ``tf_agents.agents.sac.sac_agent``:

    1. To reduce computation, here we sample actions only once for calculating
    actor, critic, and alpha loss while ``tf_agents.agents.sac.sac_agent``
    samples actions for each loss. This difference has little influence on
    the training performance.

    2. We calculate losses for every sampled steps.
    :math:`(s_t, a_t), (s_{t+1}, a_{t+1})` in sampled transition are used
    to calculate actor, critic and alpha loss while
    ``tf_agents.agents.sac.sac_agent`` only uses :math:`(s_t, a_t)` and
    critic loss for :math:`s_{t+1}` is 0. You should handle this carefully,
    it is equivalent to applying a coefficient of 0.5 on the critic loss.

    3. We mask out ``StepType.LAST`` steps when calculating losses but
    ``tf_agents.agents.sac.sac_agent`` does not. We believe the correct
    implementation should mask out ``LAST`` steps. And this may make different
    performance on same tasks.

    In addition to continuous actions addressed by the original paper, this
    algorithm also supports discrete actions and a mixture of discrete and
    continuous actions. The networks for computing Q values :math:`Q(s,a)` and
    sampling acitons can be divided into 3 cases according to action types:

    1. Discrete only: a ``QNetwork`` is used for estimating Q values. There will
       be no actor network to learn because actions can be directly sampled from
       the Q values: :math:`p(a|s) \propto \exp(\frac{Q(s,a)}{\alpha})`.
    2. Continuous only: a ``CriticNetwork`` is used for estimating Q values. An
       ``ActorDistributionNetwork`` for sampling actions will be learned according
       to Q values.

    In addition to the entropy regularization described in the SAC paper, we
    also support KL-Divergence regularization if a prior actor is provided.
    In this case, the training objective is:
        :math:`E_\pi(\sum_t \gamma^t(r_t - \alpha D_{\rm KL}(\pi(\cdot)|s_t)||\pi^0(\cdot)|s_t)))`
    where :math:`pi^0` is the prior actor.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 q_network_cls=QNetwork,
                 reward_weights=None,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 calculate_priority=False,
                 num_critic_replicas=2,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 prior_actor_ctor=None,
                 target_kld_per_dim=3.,
                 initial_log_alpha=0.0,
                 max_log_alpha=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 debug_summaries=False,
                 reproduce_locomotion=False,
                 name="SacAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network will be called
                to sample continuous actions. All of its output specs must be
                continuous. Note that we don't need a discrete actor network
                because a discrete action can simply be sampled from the Q values.
            critic_network_cls (Callable): is used to construct critic network.
                for estimating ``Q(s,a)`` given that the action is continuous.
            q_network (Callable): is used to construct QNetwork for estimating ``Q(s,a)``
                given that the action is discrete. Its output spec must be consistent with
                the discrete action in ``action_spec``.
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the q values is used.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``alf.get_config_value(TrainerConfig.epsilon_greedy)``
            use_entropy_reward (bool): whether to include entropy as reward
            calculate_priority (bool): whether to calculate priority. This is
                only useful if priority replay is enabled.
            num_critic_replicas (int): number of critics to be used. Default is 2.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            initial_log_alpha (float): initial value for variable ``log_alpha``.
            max_log_alpha (float|None): if not None, ``log_alpha`` will be
                capped at this value.
            target_entropy (float|Callable|None): If a floating value, it's the
                target average policy entropy, for updating ``alpha``. If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. If ``None``, a default entropy will
                be calculated.
            prior_actor_ctor (Callable): If provided, it will be called using
                ``prior_actor_ctor(observation_spec, action_spec, debug_summaries=debug_summaries)``
                to constructor a prior actor. The output of the prior actor is
                the distribution of the next action. Two prior actors are implemented:
                ``alf.algorithms.prior_actor.SameActionPriorActor`` and
                ``alf.algorithms.prior_actor.UniformPriorActor``.
            target_kld_per_dim (float): ``alpha`` is dynamically adjusted so that
                the KLD is about ``target_kld_per_dim * dim``.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between
                ``[-dqda_clipping, dqda_clipping]``. Will not perform clipping if
                ``dqda_clipping == 0``.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        self._calculate_priority = calculate_priority
        self._use_entropy_reward = use_entropy_reward
        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value('TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy

        self._num_critic_replicas = num_critic_replicas
        critic_networks, actor_network, self._act_type = self._make_networks(
            observation_spec, action_spec, reward_spec, actor_network_cls,
            critic_network_cls, q_network_cls)

        action_state_spec = SacActionState(
            actor_network=(() if self._act_type == ActionType.Discrete else
                           actor_network.state_spec),
            critic=(() if self._act_type == ActionType.Continuous else
                    critic_networks.state_spec))
        train_state_spec = SacState(
            action=action_state_spec,
            actor=(() if self._act_type != ActionType.Continuous else
                   critic_networks.state_spec),
            critic=SacCriticState(
                critics=critic_networks.state_spec,
                target_critics=critic_networks.state_spec))
        predict_state_spec = SacState(action=action_state_spec)

        # % 继承
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            predict_state_spec=predict_state_spec,
            reward_weights=reward_weights,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        log_alpha = nn.Parameter(torch.tensor(float(initial_log_alpha)))
        if actor_optimizer is not None and actor_network is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, nest.flatten(log_alpha))

        self._log_alpha = log_alpha
        if max_log_alpha is not None:
            self._max_log_alpha = torch.tensor(float(max_log_alpha))
        else:
            self._max_log_alpha = None

        self._actor_network = actor_network
        self._critic_networks = critic_networks
        self._target_critic_networks = self._critic_networks.copy(name='target_critic_networks')
        self._dqda_clipping = dqda_clipping

        td_lambda = 0.95 if config.mini_batch_length > 2 else 0
        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(critic_loss_ctor(td_lambda=td_lambda,
                                                        name="critic_loss%d" % (i + 1)))

        self._target_entropy = _set_target_entropy(
            self.name, target_entropy, nest.flatten(self._action_spec))

        self._training_started = False
        self._reproduce_locomotion = reproduce_locomotion

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       continuous_actor_network_cls, critic_network_cls, q_network_cls):
        discrete_action_spec = [spec for spec in nest.flatten(action_spec) if spec.is_discrete]
        continuous_action_spec = [spec for spec in nest.flatten(action_spec) if spec.is_continuous]

        if discrete_action_spec:
            discrete_action_spec = action_spec
        elif continuous_action_spec:
            continuous_action_spec = action_spec

        actor_network = None
        if continuous_action_spec:
            assert continuous_actor_network_cls is not None, (
                "If there are continuous actions, then a ActorDistributionNetwork "
                "must be provided for sampling continuous actions!")
            actor_network = continuous_actor_network_cls(
                input_tensor_spec=observation_spec, action_spec=continuous_action_spec)
            act_type = ActionType.Continuous
            assert critic_network_cls is not None, (
                "If only continuous actions exist, then a CriticNetwork must be provided!")
            critic_network = critic_network_cls(
                input_tensor_spec=(observation_spec, continuous_action_spec))
            critic_networks = critic_network.make_parallel(self._num_critic_replicas)

        if discrete_action_spec:
            act_type = ActionType.Discrete
            assert len(alf.nest.flatten(discrete_action_spec)) == 1, (
                "Only support at most one discrete action currently! "
                "Discrete action spec: {}".format(discrete_action_spec))
            assert q_network_cls is not None, (
                "If there exists a discrete action, then QNetwork must "
                "be provided!")
            q_network = q_network_cls(
                input_tensor_spec=observation_spec, action_spec=action_spec)
            critic_networks = q_network.make_parallel(self._num_critic_replicas)

        return critic_networks, actor_network, act_type

    # % predict_step & rollout_step
    def predict_step(self, inputs: TimeStep, state: SacState):
        action_dist, action, _, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=self._epsilon_greedy,  # 0.
            eps_greedy_sampling=True)
        return AlgStep(
            output=action,
            state=SacState(action=action_state),
            info=SacInfo(action_distribution=action_dist))

    def rollout_step(self, inputs: TimeStep, state: SacState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        action_dist, action, _, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=1.0,
            eps_greedy_sampling=True,
            rollout=True)

        if self.need_full_rollout_state():
            _, critics_state = self._compute_critics(
                self._critic_networks, inputs.observation, action,
                state.critic.critics)
            _, target_critics_state = self._compute_critics(
                self._target_critic_networks, inputs.observation, action,
                state.critic.target_critics)
            critic_state = SacCriticState(
                critics=critics_state, target_critics=target_critics_state)
            if self._act_type == ActionType.Continuous:
                # During unroll, the computations of ``critics_state`` and
                # ``actor_state`` are the same.
                actor_state = critics_state
            else:
                actor_state = ()
        else:
            actor_state = state.actor
            critic_state = state.critic

        new_state = SacState(
            action=action_state, actor=actor_state, critic=critic_state)
        return AlgStep(
            output=action,
            state=new_state,
            info=SacInfo(action=action, action_distribution=action_dist))

    def _predict_action(self, observation, state: SacActionState,
                        epsilon_greedy=None, eps_greedy_sampling=False,
                        rollout=False):
        new_state = SacActionState()
        if self._act_type == ActionType.Continuous:
            continuous_action_dist, actor_network_state = self._actor_network(
                observation, state=state.actor_network)
            new_state = new_state._replace(actor_network=actor_network_state)
            if eps_greedy_sampling:  # predict 0. & rollout 1.
                continuous_action = dist_utils.epsilon_greedy_sample(continuous_action_dist, epsilon_greedy)
            else:  # train
                continuous_action = dist_utils.rsample_action_distribution(continuous_action_dist)

            action_dist = continuous_action_dist
            action = continuous_action

        q_values = None
        critic_network_inputs = (observation, None)
        if self._act_type == ActionType.Discrete:
            q_values, critic_state = self._compute_critics(
                self._critic_networks, *critic_network_inputs, state.critic)

            new_state = new_state._replace(critic=critic_state)
            alpha = torch.exp(self._log_alpha).detach()
            logits = q_values / alpha  # p(a|s) = exp(Q(s,a)/alpha) / Z;
            discrete_action_dist = td.Categorical(logits=logits)
            if eps_greedy_sampling:  # predict & rollout ε=0
                discrete_action = dist_utils.epsilon_greedy_sample(discrete_action_dist, epsilon_greedy)
            else:  # train ε=1
                discrete_action = dist_utils.sample_action_distribution(discrete_action_dist)

            action_dist = discrete_action_dist
            action = discrete_action

        if (self._reproduce_locomotion and rollout and not self._training_started):
            # This uniform sampling seems important because for a squashed Gaussian,
            # even with a large scale, a random policy is not nearly uniform.
            action = alf.nest.map_structure(
                lambda spec: spec.sample(outer_dims=observation.shape[:1]),
                self._action_spec)

        return action_dist, action, q_values, new_state

    def _compute_critics(self, critic_net, observation, action, critics_state,
                         replica_min=True):
        if self._act_type == ActionType.Continuous:
            observation = (observation, action)

        # discrete: critics shape [B, replicas, num_actions]
        # continuous: critics shape [B, replicas]
        critics, critics_state = critic_net(observation, state=critics_state)
        if replica_min:
            critics = critics.min(dim=1)[0]

        return critics, critics_state

    # % train_step()
    def train_step(self, inputs: TimeStep, state: SacState, rollout_info: SacInfo):
        self._training_started = True

        action_distribution, action, critics, action_state = self._predict_action(
            inputs.observation, state=state.action)

        # % 计算熵
        log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a), action_distribution, action)
        log_pi = sum(nest.flatten(log_pi))  # Σπ'^logπ'^

        # % 计算损失
        actor_state, actor_loss = self._actor_train_step(
            inputs, state.actor, action, critics, log_pi, action_distribution)
        critic_state, critic_info = self._critic_train_step(
            inputs, state.critic, rollout_info, action, action_distribution)
        alpha_loss = self._alpha_train_step(log_pi)

        state = SacState(action=action_state, actor=actor_state, critic=critic_state)
        info = SacInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            action_distribution=action_distribution,
            actor=actor_loss,
            critic=critic_info,
            alpha=alpha_loss,
            log_pi=log_pi)
        return AlgStep(action, state, info)

    def _select_q_value(self, action, q_values):
        """Use ``action`` to index and select Q values.
        Args:
            action (Tensor): discrete actions with shape ``[batch_size]``.
            q_values (Tensor): Q values with shape ``[batch_size, replicas, num_actions]``,
                replicas:
        Returns:
            Tensor: selected Q values with shape ``[batch_size, replicas]``.
        """
        # [batch_size] -> [batch_size, 1, 1]
        action = action.view(q_values.shape[0], 1, 1)
        # [batch_size, 1, 1] -> [batch_size, n, 1]
        action = action.expand(-1, q_values.shape[1], -1).long()
        return q_values.gather(2, action).squeeze(2)

    def _actor_train_step(self, inputs: TimeStep, state, action, critics,
                          log_pi, action_distribution):
        """Q - αlogπθ(a|s)"""
        neg_entropy = sum(nest.flatten(log_pi))

        if self._act_type == ActionType.Discrete:
            # Pure discrete case doesn't need to learn an actor network
            return (), LossInfo(extra=SacActorInfo(neg_entropy=neg_entropy))
        if self._act_type == ActionType.Continuous:
            q_value, critics_state = self._compute_critics(
                self._critic_networks, inputs.observation, action, state)
            cont_alpha = torch.exp(self._log_alpha).detach()

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping, self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss((dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        # This sum() will reduce all dims so q_value can be any rank
        dqda = nest_utils.grad(action, q_value.sum())
        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))
        actor_loss += cont_alpha * log_pi

        actor_info = LossInfo(
            loss=actor_loss,
            extra=SacActorInfo(actor_loss=actor_loss, neg_entropy=neg_entropy))
        return critics_state, actor_info

    def _critic_train_step(self, inputs: TimeStep, state: SacCriticState,
                           rollout_info: SacInfo, action, action_distribution):
        critics, critics_state = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_info.action,
            state.critics,
            replica_min=False)
        target_critics, target_critics_state = self._compute_critics(
            self._target_critic_networks,
            inputs.observation,
            action,
            state.target_critics)

        if self._act_type == ActionType.Discrete:
            critics = self._select_q_value(rollout_info.action, critics)
            probs = action_distribution.probs  # [B]
            target_critics = torch.sum(probs * target_critics, dim=1)  # V = ∑π(a|s) * Q(s,a)

        target_critic = target_critics.detach()

        state = SacCriticState(critics=critics_state, target_critics=target_critics_state)
        info = SacCriticInfo(critics=critics, target_critic=target_critic)

        return state, info

    def _alpha_train_step(self, log_pi):
        alpha_loss = nest.map_structure(lambda la, lp, t: la * (-lp - t).detach(),
                                        self._log_alpha, log_pi, self._target_entropy)
        return sum(nest.flatten(alpha_loss))

    # % calc_loss
    def calc_loss(self, info: SacInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_loss = info.actor

        if self._reproduce_locomotion:
            policy_l = math_ops.add_ignore_empty(actor_loss.loss, alpha_loss)
            policy_mask = torch.ones_like(policy_l)
            policy_mask[0, :] = 0.
            critic_l = critic_loss.loss
            critic_mask = torch.ones_like(critic_l)
            critic_mask[-1, :] = 0.
            loss = critic_l * critic_mask + policy_l * policy_mask
        else:
            loss = math_ops.add_ignore_empty(actor_loss.loss, critic_loss.loss + alpha_loss)

        extra = SacLossInfo(actor=actor_loss.extra, critic=critic_loss.extra, alpha=alpha_loss)
        return LossInfo(loss=loss, priority=critic_loss.priority, extra=extra)

    def _calc_critic_loss(self, info: SacInfo):
        """Put entropy reward in ``experience.reward`` instead of ``target_critics`` 、
        because in multi-step TD learning, the entropy should also appear in intermediate steps!
        This doesn't affect one-step TD loss, however."""
        if self._use_entropy_reward:
            with torch.no_grad():
                entropy_reward = nest.map_structure(
                    lambda la, lp: -torch.exp(la) * lp, self._log_alpha, info.log_pi)
                entropy_reward = sum(nest.flatten(entropy_reward))
                discount = self._critic_losses[0].gamma * info.discount
                info = info._replace(reward=info.reward + entropy_reward * discount)

        critic_info = info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            critic_losses.append(
                l(info=info,
                  value=critic_info.critics[:, :, i, ...],
                  target_value=critic_info.target_critic).loss)
        critic_loss = math_ops.add_n(critic_losses)

        if self._calculate_priority:
            valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = ((critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        return LossInfo(
            loss=critic_loss,
            priority=priority,
            extra=critic_loss / float(self._num_critic_replicas))

    # % after_update
    def after_update(self, root_inputs, info: SacInfo):
        self._update_target()
        if self._max_log_alpha is not None:
            nest.map_structure(
                lambda la: la.data.copy_(torch.min(la, self._max_log_alpha)),
                self._log_alpha)

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']
