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
    "TaacActorInfo", ["actor_loss", "b1_a_entropy", "beta_entropy"], default_value=())

TaacCriticInfo = namedtuple(
    "TaacCriticInfo", ["critics", "target_critic", "baseline"], default_value=())

TaacInfo = namedtuple(
    "TaacInfo", [
        "reward", "step_type", "action", "prev_action", "discount",
        "action_distribution", "rollout_b", "b", "actor", "critic", "alpha", "option",
        "repeats"],
    default_value=())

TaacLossInfo = namedtuple('TaacLossInfo', ["actor", "critic", "option", "alpha"])

# % 其他,相当于提前声明了列表变量,方便打包
Distributions = namedtuple("Distributions", ["beta_dist", "b1_action_dist"])  # β=Q^/Q-+Q^, π(a^|)

ActPredOutput = namedtuple("ActPredOutput", ["dists", "b", "actions"], default_value=())

Mode = Enum('AlgorithmMode', ('predict', 'rollout', 'train'))

@alf.configurable
class TaacoAlgorithm(OffPolicyAlgorithm):
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
                 option_network_cls=QNetwork,
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
                 option_optimizer=None,
                 alpha_optimizer=None,
                 termination_reg=0.01,
                 debug_summaries=False,
                 randomize_first_state_action=False,
                 b1_advantage_clipping=None,
                 max_repeat_steps=None,
                 target_entropy=None,
                 name="TaacoAlgorithm"):
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
            option_network_cls (Callable): s -> Q, Q[b0_action] ～ b0,b1.
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
            option_optimizer (torch.optim.optimizer): The optimizer for option.
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
        critic_networks, actor_network, option_network = self._make_networks_impl(
            observation_spec, action_spec,
            actor_network_cls, critic_network_cls, q_network_cls, option_network_cls)

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
        if actor_optimizer is not None and actor_network is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if option_optimizer is not None:
            self.add_optimizer(option_optimizer, [option_network])
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
        self._option_network = option_network
        self._termination_reg = termination_reg
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
    def _make_networks_impl(self, observation_spec, action_spec,
                            actor_network_cls, critic_network_cls, q_network_cls, option_network_cls):
        actor_network = None
        q_network = q_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        critic_networks = q_network.make_parallel(self._num_critic_replicas)
        option_network = option_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        return critic_networks, actor_network, option_network

    def _compute_beta_and_action(self, observation, state, epsilon_greedy, mode):
        # % 1. sac_base: a^ ~ π(a^|s,a-)
        q_values = self._compute_critics(self._critic_networks, observation)

        q_alpha = torch.exp(self._log_alpha[1]).detach()
        q_logits = q_values / q_alpha  # p(a|s) = Q(s,a)/alpha
        b1_action_dist = td.Categorical(logits=q_logits)
        if mode == Mode.predict:  # * a^ ～ a^_dist
            b1_action = dist_utils.epsilon_greedy_sample(b1_action_dist, epsilon_greedy)
        else:
            b1_action = dist_utils.sample_action_distribution(b1_action_dist)  # 随机采样

        # % TODO 之后可以尝试一下双option网络 & (s,a-)->β
        # % 2. 计算β分布
        b0_action = state.action  # a- 列索引
        b_values, _ = self._option_network(observation)
        length = torch.linspace(0, b_values.shape[0] - 1, b_values.shape[0]).long()  # 行索引
        b0_value = b_values[length, b0_action]  # b0的中断价值 Qo(s,b0)

        beta_alpha = torch.exp(self._log_alpha[0]).detach()
        b_logits = b0_value / beta_alpha  # p(b0|s) = Qo(s,b0)/alpha
        b_probs = b_logits.sigmoid()
        beta_dist = td.Categorical(probs=torch.stack([1 - b_probs, b_probs], 1))  # b0, b1

        # % 3. 判断是否选择a^
        if mode == Mode.predict:  # * b ～ β, [选a-的概率,选a^的概率]
            b = dist_utils.epsilon_greedy_sample(beta_dist, epsilon_greedy)
        else:
            b = dist_utils.sample_action_distribution(beta_dist)

        dists = Distributions(beta_dist=beta_dist, b1_action_dist=b1_action_dist)
        return ActPredOutput(
            dists=dists,  # β, π(a^|s,a-)
            b=b,  # prob(a^)
            actions=(b0_action, b1_action))  # a-, a^

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
            action=new_state.action,  # a, real action for critic training
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
        new_state = conditional_update(  # a-/a^ -> a 得到当前实际执行的aticon
            target=state,
            cond=condition,
            func=_b1_action,
            b1_action=b1_action,
            state=state)

        new_state = new_state._replace(repeats=new_state.repeats + 1)  # 记录重复动作数
        return ap_out, new_state

    def _compute_critics(self, critic_net, observation,
                         replica_min=True):
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

        # TODO beta_dist不知道要不要改
        # % 计算 action(高斯分布)和β(分类分布) 熵
        b1_a_entropy = -dist_utils.compute_log_probability(b1_action_dist, b1_action)  # -Σπ'^logπ'^
        beta_entropy = beta_dist.entropy()  # -Σβlogβ 高斯分布的处理要简单一些

        # % 计算损失
        actor_loss = self._actor_train_step()
        critic_info = self._critic_train_step(
            inputs, rollout_info.action, b0_action, beta_dist, b1_action_dist)
        option_loss = self._option_train_step(
            inputs, beta_dist, b1_action_dist, critic_info.critics, critic_info.baseline)
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
            option=option_loss,
            b=ap_out.b,  # b~_n
            alpha=alpha_loss,
            repeats=state.repeats)
        return AlgStep(output=action, state=new_state, info=info)

    def _select_q_value(self, action, q_values):
        """Use ``action`` to index and select Q values.
        Args:
            action (Tensor): discrete actions with shape ``[batch_size]``.
            q_values (Tensor): Q values with shape ``[batch_size, replicas, num_actions]``,
        Returns:
            Tensor: selected Q values with shape ``[batch_size, replicas]``.
        """
        # [batch_size] -> [batch_size, 1, 1]
        action = action.view(q_values.shape[0], 1, 1)
        # [batch_size, 1, 1] -> [batch_size, n, 1]
        action = action.expand(-1, q_values.shape[1], -1).long()
        return q_values.gather(2, action).squeeze(2)

    def _actor_train_step(self):
        return LossInfo()

    def _critic_train_step(self, inputs: TimeStep, rollout_action, b0_action, beta_dist, b1_action_dist):
        """compute target_q"""
        with torch.no_grad():
            target_critic = self._compute_critics(  # minQ'(s,∑a)
                self._target_critic_networks,
                inputs.observation)

            # % TODO 可能出错的地方
            # % 目标价值  V' = (1-b)Q'(s,a-) + b V(s)
            beta_probs = beta_dist.probs  # b, 2维
            action_probs = b1_action_dist.probs  # action_b, n维

            length = torch.linspace(0, target_critic.shape[0] - 1, target_critic.shape[0]).long()
            b0_probs = action_probs[length, b0_action]
            b0_target_critic = target_critic[length, b0_action]

            target_critic = beta_probs[..., 0] * b0_probs * b0_target_critic + \
                            beta_probs[..., 1] * torch.sum(action_probs * target_critic, dim=1)  # targte_V = ∑π*target_Q

        critics = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            replica_min=False)
        baseline = torch.sum(action_probs * critics.min(dim=1)[0], dim=1)  # V = ∑πQ
        critics = self._select_q_value(rollout_action, critics)

        return TaacCriticInfo(critics=critics, target_critic=target_critic, baseline=baseline)

    def _alpha_train_step(self, beta_entropy, action_entropy):
        """α * -πlogπ"""
        alpha_loss = (self._log_alpha[1] * (action_entropy - self._target_entropy[1]).detach())
        alpha_loss += (self._log_alpha[0] * (beta_entropy - self._target_entropy[0]).detach())
        return alpha_loss

    def _option_train_step(self, inputs, beta_dist, b1_action_dist, critics, baseline):
        """Termination loss   β(b1) * (Q(s,a^) - maxQ(s) + reg) * (1-done)"""
        is_firsts = (inputs.step_type == StepType.FIRST).to(dtype=torch.float32)
        is_lasts = (inputs.step_type == StepType.LAST).to(dtype=torch.float32)

        critic = critics.min(dim=1)[0]
        option_loss = beta_dist.probs[:, 1] * (critic.detach() - baseline.detach() + self._termination_reg)
        option_loss = option_loss * (1 - is_lasts) * (1 - is_firsts)

        return option_loss

    # % calc_loss
    def calc_loss(self, info: TaacInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_loss = info.actor
        option_loss = info.option

        with alf.summary.scope(self._name):  # 记录重复数
            alf.summary.scalar("train_repeats/mean", torch.mean(info.repeats.to(torch.float32)))

        loss = math_ops.add_ignore_empty(actor_loss.loss, alpha_loss + critic_loss.loss + option_loss)
        extra = TaacLossInfo(actor=actor_loss.extra, critic=critic_loss.extra, option=option_loss, alpha=alpha_loss)
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
