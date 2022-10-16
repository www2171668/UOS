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
from alf.networks import QNetwork, QRNNNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, dist_utils, losses, math_ops, tensor_utils
from alf.utils.conditional_ops import conditional_update
from alf.utils.summary_utils import safe_mean_hist_summary

# * 因为没有用RNN,所以state都为空
TaacState = namedtuple("TaacState", ["action", "repeats"], default_value=())  # a- a-重复数

TaacActorInfo = namedtuple(
    "TaacActorInfo", ["actor_loss", "b1_a_entropy", "beta_entropy", "value_loss"], default_value=())

TaacCriticInfo = namedtuple(
    "TaacCriticInfo", ["critics", "target_critic", "value_loss"], default_value=())

TaacInfo = namedtuple(
    "TaacInfo", [
        "reward", "step_type", "action", "prev_action", "discount",
        "action_distribution", "rollout_b", "b", "actor", "critic", "alpha",
        "repeats"], default_value=())

TaacLossInfo = namedtuple('TaacLossInfo', ('actor', 'critic', 'alpha'))

# % 其他,相当于提前声明了列表变量,方便打包
Distributions = namedtuple("Distributions", ["beta_dist", "b1_action_dist"])  # β=Q^/Q-+Q^, π(a^|)

ActPredOutput = namedtuple("ActPredOutput", ["dists", "b", "actions", "q_values2"], default_value=())

Mode = Enum('AlgorithmMode', ('predict', 'rollout', 'train'))

@alf.configurable
class TaacdAlgorithm(OffPolicyAlgorithm):
    r"""Temporally abstract actor-critic algorithm.

    In a nutsell, for inference TAAC adds a second stage that chooses between a
    candidate trajectory :math:`\hat{\action}` output by an SAC actor and the previous
    trajectory :math:`\action^-`. 对于1-步TD,action是a-

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
                 q_network_cls=QNetwork,
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
                 name="TaacdAlgorithm"):
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
                sac使用的是OneStepTDLoss
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
        # % 构建网络
        self._action_spec = action_spec
        self._num_critic_replicas = num_critic_replicas
        critic_networks, actor_network = self._make_networks_impl(
            observation_spec, action_spec, reward_spec,
            actor_network_cls, critic_network_cls, q_network_cls)

        # % 训练数据规格
        train_state_spec = TaacState(
            action=self._action_spec,
            repeats=TensorSpec(shape=(), dtype=torch.int64))

        # % 继承
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

        # % 优化器 √
        log_alpha = (nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(())))
        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, list(log_alpha))

        # % 必要参数
        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value('TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy
        self._log_alpha = log_alpha
        self._log_alpha_paralist = nn.ParameterList(list(log_alpha))
        self._actor_network = actor_network
        self._critic_networks = critic_networks
        self._target_critic_networks = self._critic_networks.copy(name='target_critic_networks')
        self.register_buffer("_training_started", torch.zeros((), dtype=torch.bool))  # 判定train_step

        # % cirtic网络
        if critic_loss_ctor is None:
            critic_loss_ctor = TAACTDLoss
        critic_loss_ctor = functools.partial(critic_loss_ctor, debug_summaries=debug_summaries)

        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(critic_loss_ctor(name="critic_loss%d" % (i + 1)))
        self._gamma = self._critic_losses[0]._gamma

        # % target entropies 初始值 for b_dist and a_dist
        self._b_spec = BoundedTensorSpec(shape=(), dtype='int64', maximum=1)  # 选择a^的概率
        if not isinstance(target_entropy, tuple):
            target_entropy = (target_entropy,) * 2
        self._target_entropy = nest.map_structure(
            lambda t, spec: _set_target_entropy(self.name, t, [spec]),
            target_entropy, (self._b_spec, action_spec))

        # % 目标网络更新
        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

    # % 创建actor和cirtic网络
    def _make_networks_impl(self, observation_spec, action_spec, reward_spec,
                            actor_network_cls, critic_network_cls, q_network_cls):
        actor_network = None
        discrete_action_spec = action_spec
        q_network = q_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        critic_networks = q_network.make_parallel(self._num_critic_replicas)

        return critic_networks, actor_network

    # % 计算β分布
    def _build_beta_dist(self, q_values2):
        """compute β (dist) *conditioned* on ``action``

           A stable implementation of categorical distribution :math:`exp(\frac{Q}{\alpha})`.  exp(Q/a)
           通过logits-max(logits)将最大概率Q值置为0,使采样只采到该Q对应的标签(另一个是负数)
        """
        with torch.no_grad():
            beta_alpha = self._log_alpha[0].exp().detach()  # β温度系数
            q_values2 = q_values2 / torch.clamp(beta_alpha, min=1e-10)
            q_values2 = q_values2 - torch.max(q_values2, dim=-1, keepdim=True)[0]
            beta_dist = td.Categorical(logits=q_values2)  # 基于[q1,q2]构建分类分布(β分布)
        return beta_dist

    def _compute_beta_and_action(self, observation, state, epsilon_greedy, mode):
        # % 1. sac_base: a^ ~ π(a^|s,a-)
        critic_network_inputs = (observation, None)
        q_values = self._compute_critics(
            self._critic_networks, *critic_network_inputs)

        alpha = torch.exp(self._log_alpha[0]).detach()
        logits = q_values / alpha  # p(a|s) = exp(Q(s,a)/alpha) / Z;
        b1_action_dist = td.Categorical(logits=logits)
        if mode == Mode.predict:
            b1_action = dist_utils.epsilon_greedy_sample(b1_action_dist, epsilon_greedy)
        else:
            b1_action = dist_utils.sample_action_distribution(b1_action_dist)

        # % 2. taac: 计算Q-和Q
        """Update the current trajectory ``action`` by moving one step ahead."""
        """Compute a new trajectory ``action`` given a new action."""
        b0_action = state.action  # a-

        with torch.no_grad():  # 旧的Q不纳入梯度计算
            q_0 = self._compute_critics(self._critic_networks, observation, b0_action)  # Q(s, a-)
            q_0 = torch.max(q_0,dim=1)[0]
        q_1 = self._compute_critics(self._critic_networks, observation, b1_action)  # Q(s, a^)
        q_1 = torch.max(q_1,dim=1)[0]

        q_values2 = torch.stack([q_0, q_1], dim=-1)  # 叠加[[q-,q],[q-,q],...]
        beta_dist = self._build_beta_dist(q_values2)  # [Q-,Q] -> β

        # % 3. 判断是否选择a^
        if mode == Mode.predict:  # * b ～ β, [a-的概率,a^的概率] -> [0] or [1]
            b = dist_utils.epsilon_greedy_sample(beta_dist, epsilon_greedy)
        else:
            b = dist_utils.sample_action_distribution(beta_dist)

        dists = Distributions(beta_dist=beta_dist, b1_action_dist=b1_action_dist)
        return ActPredOutput(
            dists=dists,  # β, π(a^|s,a-)
            b=b,  # prob(a^)
            actions=(b0_action, b1_action),  # a-, a^
            q_values2=q_values2)  # [Q-,Q]

    # % 基礎
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
            action=new_state.action,  # a^, for critic training
            b=ap_out.b,  # b, for Info
            repeats=state.repeats)
        return AlgStep(output=new_state.action, state=new_state, info=info)

    def _predict_action(self, observation, state, epsilon_greedy=None, mode=Mode.rollout):
        """selectively update with new actions"""
        ap_out = self._compute_beta_and_action(observation, state, epsilon_greedy, mode)

        # % 训练之前的随机行走过程
        if not common.is_eval() and not self._training_started:
            b = self._b_spec.sample(observation.shape[:1])  # b
            b1_action = self._action_spec.sample(observation.shape[:1])  # a^
            ap_out = ap_out._replace(b=b, actions=(ap_out.actions[0], b1_action))  # 替换

        # % 更新action缓存
        def _b1_action(b1_action, state):
            """选择新动作时，action更新为a^，repeats清0"""
            new_state = state._replace(action=b1_action, repeats=torch.zeros_like(state.repeats))
            return new_state

        b0_action, b1_action = ap_out.actions  # a-, a^
        condition = ap_out.b.to(torch.bool)  # a^ -> True
        new_state = conditional_update(
            target=state,
            cond=condition,
            func=_b1_action,
            b1_action=b1_action,
            state=state)

        new_state = new_state._replace(repeats=new_state.repeats + 1)  # 记录重复动作数
        return ap_out, new_state

    def _compute_critics(self, critic_net, observation, action,
                         replica_min=True, apply_reward_weights=True):
        """Compute Q(s,a)   这里的action就是action"""
        critics, _ = critic_net(observation)  # [B, replicas]
        if replica_min:
            critics = critics.min(dim=1)[0]

        return critics

    # % train_step()
    def train_step(self, inputs: TimeStep, state: TaacState, rollout_info: TaacInfo):
        self._training_started.fill_(True)
        ap_out, new_state = self._predict_action(inputs.observation, state=state, mode=Mode.train)

        beta_dist = ap_out.dists.beta_dist  # β
        b1_action_dist = ap_out.dists.b1_action_dist  # π(a'^|s,a)
        b0_action, b1_action = ap_out.actions  # a-  a^
        q_values2 = ap_out.q_values2  # q2

        # % 计算 action(高斯分布)和β(分类分布) 熵
        b1_a_entropy = -dist_utils.compute_log_probability(b1_action_dist, b1_action)  # -Σπ'^logπ'^
        beta_entropy = beta_dist.entropy()  # -Σβlogβ 高斯分布的处理要简单一些

        # % 计算损失
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
            rollout_b=rollout_info.b,  # b_n
            action_distribution=ap_out.dists,
            actor=actor_loss,
            critic=critic_info,
            b=ap_out.b,  # b~_n
            alpha=alpha_loss,
            repeats=state.repeats)
        return AlgStep(output=action, state=new_state, info=info)

    def _select_q_value(self, action, q_values):
        """Use ``action`` to index and select Q values.
        Args:
            action (Tensor): discrete actions with shape ``[batch_size]``.
            q_values (Tensor): Q values with shape
                ``[batch_size, replicas, num_actions, reward_dim]``, where
                ``reward_dim`` is optional for multi-dim reward.
        Returns:
            Tensor: selected Q values with shape
                ``[batch_size, replicas, reward_dim]``.
        """
        # [batch_size] -> [batch_size, 1, 1, ...]
        action = action.view(q_values.shape[0], 1, 1)
        # [batch_size, 1, 1, ...] -> [batch_size, n, 1, reward_dim]
        action = action.expand(-1, q_values.shape[1], -1).long()
        return q_values.gather(2, action).squeeze(2)

    def _actor_train_step(self, a, b1_a_entropy, beta_dist, beta_entropy, q_values2):
        return ()

    def _critic_train_step(self, inputs: TimeStep, rollout_action, b0_action, b1_action, beta_dist):
        """compute target_q"""
        with torch.no_grad():
            target_q_0 = self._compute_critics(  # Q'(s,a-)
                self._target_critic_networks,
                inputs.observation,
                b0_action,
                apply_reward_weights=False)
            target_q_1 = self._compute_critics(  # Q'(s,a^)
                self._target_critic_networks,
                inputs.observation,
                b1_action,
                apply_reward_weights=False)

            # % TODO 这里好像不好做，需要有新的公式
            # 提取b对应的目标价值  Q' = (1-b)Q'(s,a-) + bQ(s,a^)
            beta_probs = beta_dist.probs
            target_critic = torch.sum(beta_probs[..., 0] * target_q_0 + beta_probs[..., 1] * target_q_1, dim=1)

        critics = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_action,
            replica_min=False,
            apply_reward_weights=False)
        critics = self._select_q_value(rollout_action, critics)

        return TaacCriticInfo(critics=critics, target_critic=target_critic)

    def _alpha_train_step(self, beta_entropy, action_entropy):
        """α * -πlogπ"""
        alpha_loss = (self._log_alpha[1] * (action_entropy - self._target_entropy[1]).detach())
        alpha_loss += (self._log_alpha[0] * (beta_entropy - self._target_entropy[0]).detach())
        return alpha_loss

    # % calc_loss
    def calc_loss(self, info: TaacInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_loss = info.actor

        if self._debug_summaries:
            with alf.summary.scope(self._name):  # 记录重复数
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
        # print(root_inputs.reward[0][0])  # 种子测试
        pass

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']

    def summarize_rollout(self, experience):
        repeats = experience.rollout_info.repeats.reshape(-1)
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("rollout_repeats/mean", torch.mean(repeats.to(torch.float32)))
