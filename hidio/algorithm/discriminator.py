import alf
import torch
import copy
import numpy as np

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import TimeStep, AlgStep, namedtuple, LossInfo, StepType
from alf.networks import EncodingNetwork

from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils import math_ops, common, losses
from alf.nest.utils import NestConcat
from alf.utils.conditional_ops import conditional_update
from alf.utils.normalizers import AdaptiveNormalizer, ScalarAdaptiveNormalizer

from alf.algorithms.vae import VariationalAutoEncoder
from alf.algorithms.decoding_algorithm import DecodingAlgorithm

import hidio.algorithm.subtrajector as sub_utils

SubTrajectory = namedtuple('SubTrajectory', ["observation", "prev_action"], default_value=())

DiscriminatorState = namedtuple("DiscriminatorState", ["first_observation", "untrans_observation", "subtrajectory"],
                                default_value=())

supported_skill_types = ["state", "action", "state_difference", "action_difference", "state_action"]

def is_action_skill(skill_type):
    return "action" in skill_type

# % 子轨迹规格设定(存储轨迹信息)
def get_subtrajectory_spec(num_steps_per_skill, observation_spec, action_spec):
    observation_traj_spec = TensorSpec(shape=(num_steps_per_skill,) + observation_spec.shape)
    action_traj_spec = TensorSpec(shape=(num_steps_per_skill,) + action_spec.shape)
    return SubTrajectory(observation=observation_traj_spec, prev_action=action_traj_spec)

# % 判别器规格设定
def get_discriminator_spec(skill_type, observation_spec, action_spec):
    if is_action_skill(skill_type):
        discriminator_spec = (observation_spec, action_spec)
    else:
        discriminator_spec = observation_spec

    return discriminator_spec

@alf.configurable
class Discriminator(Algorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 skill_spec,
                 config: TrainerConfig,
                 discriminator_ctor=EncodingNetwork,
                 discriminator_activation=torch.tanh,
                 skill_encoder_ctor=None,
                 observation_transformer=math_ops.identity,
                 vae_model=False,
                 normalizer=False,
                 sub_noise=0.,
                 optimizer=None,
                 sparse_reward=False,
                 debug_summaries=False,
                 num_steps_per_skill=3,
                 skill_type="state_difference",
                 name="Discriminator"):
        """If ``sparse_reward=True``, then the discriminator will only predict at the skill switching steps."""
        if skill_spec.is_discrete:
            assert isinstance(skill_spec, BoundedTensorSpec)
            skill_dim = skill_spec.maximum - skill_spec.minimum + 1
        else:
            assert len(skill_spec.shape) == 1, "Only 1D skill vector is supported"
            skill_dim = skill_spec.shape[0]

        self._skill_spec = skill_spec
        self._skill_dim = skill_dim
        self._vae_model = vae_model

        # % 定义state_spec & 构建判别器网络
        assert skill_type in supported_skill_types, ("Skill type must be in: %s" % supported_skill_types)
        subtrajectory_spec = get_subtrajectory_spec(num_steps_per_skill, observation_spec, action_spec)
        discriminator_spec = get_discriminator_spec(skill_type, observation_spec, action_spec)
        discriminator = self._create_discriminator(discriminator_ctor, discriminator_spec, skill_type, discriminator_activation)

        train_state_spec = DiscriminatorState(
            first_observation=observation_spec,
            untrans_observation=observation_spec,  # prev untransformed observation diff for pred
            subtrajectory=subtrajectory_spec)
        predict_state_spec = DiscriminatorState(
            first_observation=observation_spec,
            subtrajectory=subtrajectory_spec)

        # % 继承
        super().__init__(
            train_state_spec=train_state_spec,
            predict_state_spec=predict_state_spec,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        # % VAE
        if vae_model:
            self._vae_encoder = VariationalAutoEncoder(
                z_dim=skill_dim,
                preprocess_network=discriminator)  # discriminator + vae

        # % 定义正则化器
        reward_adapt_speed = 8.0
        if normalizer:
            self._discriminator_normalizer = AdaptiveNormalizer(tensor_spec=discriminator_spec)
            self._reward_normalizer = ScalarAdaptiveNormalizer(speed=reward_adapt_speed)

        # % 必要参数
        self._skill_type = skill_type
        self._discriminator = discriminator
        self._normalizer = normalizer
        self._sub_noise = sub_noise

        # exp observation won't be automatically transformed when it's sampled from the replay buffer. We do this manually.
        self._observation_transformer = observation_transformer
        self._num_steps_per_skill = num_steps_per_skill

    # % 创建网络
    def _create_discriminator(self, discriminator_ctor, discriminator_spec, skill_type, activation):
        preprocessing_combiner = None
        if is_action_skill(skill_type):
            preprocessing_combiner = NestConcat()

        disc_inputs = dict(input_tensor_spec=discriminator_spec,
                           preprocessing_combiner=preprocessing_combiner)
        if not self._vae_model:
            disc_inputs["last_layer_size"] = self._skill_dim
            disc_inputs["last_activation"] = activation
        discriminator = discriminator_ctor(**disc_inputs)
        return discriminator

    # % 基础
    def predict_step(self, time_step: TimeStep, state: DiscriminatorState):
        observation, switch_skill = time_step.observation
        first_observation = sub_utils.update_state_if_necessary(switch_skill, observation, state.first_observation)
        subtrajectory = sub_utils.clear_subtrajectory_if_necessary(switch_skill, state.subtrajectory)
        new_state = DiscriminatorState(first_observation=first_observation,
                                       subtrajectory=subtrajectory)
        return AlgStep(state=new_state)

    def rollout_step(self, time_step: TimeStep, state: DiscriminatorState):
        """Update the state based on switch_skill."""
        observation, _, switch_skill, _ = time_step.observation
        first_observation = sub_utils.update_state_if_necessary(switch_skill, observation, state.first_observation)
        subtrajectory = sub_utils.clear_subtrajectory_if_necessary(switch_skill, state.subtrajectory)
        new_state = DiscriminatorState(first_observation=first_observation,
                                       untrans_observation=time_step.untransformed.observation,
                                       subtrajectory=subtrajectory)
        return AlgStep(state=new_state)

    def train_step(self, inputs: TimeStep, state: DiscriminatorState, rollout_info, trainable=True):
        """This function trains the discriminator or generates intrinsic rewards."""
        untrans_observation, prev_skill, switch_skill, steps = inputs.observation
        observation = self._observation_transformer(untrans_observation)

        loss = self._predict_skill_loss(observation, inputs.prev_action, prev_skill, steps, state)

        first_observation = sub_utils.update_state_if_necessary(switch_skill, observation, state.first_observation)
        subtrajectory = sub_utils.clear_subtrajectory_if_necessary(switch_skill, state.subtrajectory)
        new_state = DiscriminatorState(first_observation=first_observation,
                                       untrans_observation=untrans_observation,
                                       subtrajectory=subtrajectory)

        valid_masks = (inputs.step_type != StepType.FIRST).to(torch.float32)
        loss *= valid_masks

        if trainable:
            info = LossInfo(loss=loss, extra=dict(discriminator_loss=loss))
            return AlgStep(state=new_state, info=info)
        else:
            intrinsic_reward = -loss.detach() / self._skill_dim
            if self._normalizer:
                intrinsic_reward = self._reward_normalizer.normalize(intrinsic_reward)
            return AlgStep(state=common.detach(new_state), info=intrinsic_reward)

    def _predict_skill_loss(self, observation, prev_action, prev_skill, steps, state):
        """s, pre_a , u, steps:{1,2,3} , state:DiscriminatorState"""
        # % 根据技能类型构造轨迹，作为判别器网络的输入
        if self._skill_type == "action":
            subtrajectory = (state.first_observation, prev_action)
        elif self._skill_type == "action_difference":
            action_difference = prev_action - state.subtrajectory[:, 1, :]
            subtrajectory = (state.first_observation, action_difference)
        elif self._skill_type == "state_action":
            subtrajectory = (observation, prev_action)  # (s, a-)
        elif self._skill_type == "state":
            subtrajectory = observation  # s
        elif self._skill_type == "state_difference":
            subtrajectory = observation - state.untrans_observation  # s'-s

        # % 轨迹泛化
        if self._sub_noise:
            noise = torch.Tensor(1).uniform_(1 - self._sub_noise, 1 + self._sub_noise)
            subtrajectory = alf.nest.map_structure(lambda sub: torch.multiply(sub, noise), subtrajectory)

        # % 正则化
        if self._normalizer:
            subtrajectory = self._discriminator_normalizer.normalize(subtrajectory)

        # % 计算损失
        if self._vae_model:
            encoder_step = self._vae_encoder.train_step(subtrajectory)
            pred_skill = encoder_step.output  # 预测
            loss = torch.sum(math_ops.square(prev_skill - pred_skill), dim=-1)
        else:
            pred_skill, _ = self._discriminator(subtrajectory)
            if self._skill_spec.is_discrete:
                loss = torch.nn.CrossEntropyLoss(reduction='none')(input=pred_skill, target=prev_skill)
            else:
                loss = torch.sum(math_ops.square(pred_skill - prev_skill), dim=-1)
        return loss

    def calc_loss(self, train_info):
        """This is called for ``train_step(trainable=True)``."""
        loss_info = LossInfo(loss=train_info.loss, extra=train_info.extra)
        with alf.summary.scope(self._name):
            alf.summary.scalar("loss", torch.mean(train_info.loss.to(torch.float32)))
        return loss_info
