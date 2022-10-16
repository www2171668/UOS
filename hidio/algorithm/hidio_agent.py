"""Agent for integrating multiple algorithms."""

import alf
import time
import torch

import alf
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.agent_helpers import AgentHelper
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import SacAlgorithm
from .skill_generator import SkillGenerator
from .discriminator import SubTrajectory

from alf.data_structures import TimeStep, AlgStep, Experience, namedtuple, StepType, LossInfo
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.nest import transform_nest
from alf.utils import math_ops
from alf.utils.conditional_ops import conditional_update
from alf.networks.preprocessors import EmbeddingPreprocessor

import hidio.algorithm.subtrajector as sub_utils

AgentState = namedtuple("AgentState", ["rl", "skill_generator"], default_value=())

AgentInfo = namedtuple("AgentInfo", ["rl", "skill_generator", "skill_discount"], default_value=())

# % 连续技能
@alf.configurable
def create_skill_spec(num_of_skills):
    return BoundedTensorSpec((num_of_skills,), maximum=1, minimum=-1)

# % 离散技能
@alf.configurable
def create_discrete_skill_spec(num_of_skills):
    return BoundedTensorSpec((), dtype="int64", maximum=num_of_skills - 1)

# % 下层observation_spec设定
@alf.configurable
def get_low_rl_input_spec(observation_spec, action_spec, num_steps_per_skill, skill_spec):
    """Rreturn:
       rl_observation_spec： [obs_traj_spec, a_traj_spec, step_spec, skill_spec]
    """
    assert observation_spec.ndim == 1 and action_spec.ndim == 1
    concat_observation_spec = TensorSpec((num_steps_per_skill * observation_spec.shape[0],))
    concat_action_spec = TensorSpec((num_steps_per_skill * action_spec.shape[0],))
    traj_spec = SubTrajectory(observation=concat_observation_spec, prev_action=concat_action_spec)
    step_spec = BoundedTensorSpec((), dtype='int64', maximum=num_steps_per_skill)
    return alf.nest.flatten(traj_spec) + [step_spec, skill_spec]

# % 构建下层预处理器
@alf.configurable
def get_low_rl_input_preprocessors(low_rl_input_specs, embedding_dim):
    return alf.nest.map_structure(
        lambda spec: EmbeddingPreprocessor(input_tensor_spec=spec, embedding_dim=embedding_dim),
        low_rl_input_specs)

@alf.configurable
class HidioAgent(RLAlgorithm):
    """Higher-level policy proposes skills for lower-level policy to executes.
       The rewards for the former is the extrinsic rewards,
       while the rewards for the latter is the negative of a skill discrimination loss. (-logq)
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 skill_spec,
                 rl_observation_spec,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config: TrainerConfig = None,
                 skill_generator_cls=SkillGenerator,
                 rl_algorithm_cls=SacAlgorithm,
                 skill_boundary_discount=0.,
                 optimizer=None,
                 observation_transformer=math_ops.identity,
                 exp_reward_relabeling=True,
                 random_skill=False,
                 dynamic_skill=False,
                 debug_summaries=False,
                 name="AgentAlgorithm"):
        """Create an Agent
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            env (Environment): The environment to interact with.
                `env` is a batched environment, which means that it runs multiple simulations simultaneously.
                Running multiple environments in parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation.
                `env` only needs to be provided to the root `Algorithm`.
            config (TrainerConfig): config for training.
                config only needs to be provided to the algorithm which performs `train_iter()` by itself.
            rl_algorithm_cls (type): The algorithm class for learning the policy.
                Currently the Hidio agent only supports SAC.
            skill_generator_cls (Algorithm): an algorithm with output a goal vector
            optimizer (tf.optimizers.Optimizer): The optimizer for training
            observation_transformer (Callable | list[Callable]): transformation(s) applied to `time_step.observation`
            exp_reward_relabeling:
            random_skill:
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
            """
        agent_helper = AgentHelper(AgentState)

        skill_generator = skill_generator_cls(
            observation_spec=observation_spec,
            action_spec=action_spec,
            skill_spec=skill_spec,
            reward_spec=reward_spec,
            env=env,
            config=config,
            debug_summaries=debug_summaries)
        agent_helper.register_algorithm(skill_generator, "skill_generator")

        rl_algorithm = rl_algorithm_cls(
            observation_spec=rl_observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            config=config,  # set use_rollout_state
            debug_summaries=debug_summaries)
        agent_helper.register_algorithm(rl_algorithm, "rl")
        self._skill_boundary_discount = skill_boundary_discount

        # % 继承
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            optimizer=optimizer,
            is_on_policy=rl_algorithm.on_policy,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name,
            **agent_helper.state_specs())

        # % 设置必要参数
        self._rl_algorithm = rl_algorithm
        self._skill_generator = skill_generator
        self._random_skill = random_skill
        self._dynamic_skill = dynamic_skill

        self._agent_helper = agent_helper
        self._observation_transformer = observation_transformer
        self._num_steps_per_skill = skill_generator.num_steps_per_skill

    def set_path(self, path):
        super().set_path(path)
        self._agent_helper.set_path(path)

    def predict_step(self, time_step: TimeStep, state: AgentState):
        new_state = AgentState()

        time_step = transform_nest(time_step, "observation", self._observation_transformer)
        subtrajectory = sub_utils.update_subtrajectory(time_step, state.skill_generator.discriminator)

        # % skill_generator
        if self._random_skill:
            skill_step = self._skill_generator.gen_random_skill(time_step, state.skill_generator)
        else:
            skill_step = self._skill_generator.predict_step(time_step, state.skill_generator)
        new_state = new_state._replace(skill_generator=skill_step.state)

        # % low_sac
        observation = sub_utils.make_low_rl_observation(
            subtrajectory,
            skill_step.state.discriminator.first_observation,
            skill_step.output,
            skill_step.state.steps,
            skill_step.info.switch_skill,
            self._num_steps_per_skill)

        rl_step = self._rl_algorithm.predict_step(
            time_step._replace(observation=observation),
            state.rl)
        new_state = new_state._replace(rl=rl_step.state)

        return AlgStep(output=rl_step.output, state=new_state, info=())

    def rollout_step(self, time_step: TimeStep, state: AgentState):
        new_state = AgentState()
        info = AgentInfo()

        time_step = transform_nest(time_step, "observation", self._observation_transformer)
        subtrajectory = sub_utils.update_subtrajectory(time_step, state.skill_generator.discriminator)

        # % skill_generator
        skill_step = self._skill_generator.rollout_step(time_step, state.skill_generator)
        new_state = new_state._replace(skill_generator=skill_step.state)
        info = info._replace(skill_generator=skill_step.info)

        # % low_sac
        observation = sub_utils.make_low_rl_observation(  # -> (s~, a~, K-k, u)
            subtrajectory,
            skill_step.state.discriminator.first_observation,
            skill_step.output,  # skill
            skill_step.state.steps,  # k -> K-k
            skill_step.info.switch_skill,  # switch
            self._num_steps_per_skill)

        rl_step = self._rl_algorithm.rollout_step(time_step._replace(observation=observation), state.rl)
        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        # % low_sac.discount
        skill_discount = (((skill_step.state.steps == 1) & (time_step.step_type != StepType.LAST)).to(torch.float32)
                          * (1 - self._skill_boundary_discount))
        info = info._replace(skill_discount=1 - skill_discount)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def train_step(self, time_step: TimeStep, state: AgentState, rollout_info: AgentInfo):
        new_state = AgentState()
        info = AgentInfo()

        subtrajectory = sub_utils.update_subtrajectory(time_step, state.skill_generator.discriminator)
        # subtrajectory = state.skill_generator.discriminator.subtrajectory

        skill_step = self._skill_generator.train_step(
            time_step,
            state.skill_generator,
            rollout_info.skill_generator)
        new_state = new_state._replace(skill_generator=skill_step.state)
        info = info._replace(skill_generator=skill_step.info)

        # % low_sac
        time_step = transform_nest(time_step, "observation", self._observation_transformer)
        observation = sub_utils.make_low_rl_observation(
            subtrajectory,
            skill_step.state.discriminator.first_observation,
            skill_step.output,
            rollout_info.skill_generator.steps,
            rollout_info.skill_generator.switch_skill,
            self._num_steps_per_skill)

        rl_step = self._rl_algorithm.train_step(
            time_step._replace(observation=observation),
            state.rl,
            rollout_info.rl)
        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        return AlgStep(state=new_state, info=info)

    def calc_loss(self, train_info: AgentInfo):
        """low_sac.calc_step

        replace reward in `experience` with the freshly
        computed intrinsic rewards by the goal generator during `train_step`.
        """
        intrinsic_reward = train_info.skill_generator.reward
        intrinsic_train_info = train_info.rl._replace(reward=intrinsic_reward)
        train_info = train_info._replace(rl=intrinsic_train_info)

        return self._agent_helper.accumulate_loss_info([self._rl_algorithm], train_info)

    def after_update(self, experience, train_info: AgentInfo):
        """
        1. -> _rl_algorithm.after_update；  low_sac
        2. -> skill_generator.after_update """
        self._agent_helper.after_update([self._rl_algorithm, self._skill_generator], experience, train_info)

    def after_train_iter(self, experience, train_info: AgentInfo):
        """
        1. -> _rl_algorithm.after_train_iter；   low_sac
        2. -> _skill_generator.after_train_iter """
        self._agent_helper.after_train_iter([self._rl_algorithm, self._skill_generator], experience, train_info)
        # print(experience.reward[0][0])

    # % low_sac 经验处理
    def preprocess_experience(self, root_inputs: Experience, rollout_info, batch_info):
        exp = root_inputs
        exp = exp._replace(discount=exp.discount * rollout_info.skill_discount)
        return exp, rollout_info

    def summarize_rollout(self, experience):
        """First call ``RLAlgorithm.summarize_rollout()`` to summarize basic
        rollout statisics. If the rl algorithm has overridden this function,
        then also call its customized version.
        """
        super(HidioAgent, self).summarize_rollout(experience)
        if (super(HidioAgent, self).summarize_rollout.__func__ !=
                self._rl_algorithm.summarize_rollout.__func__):
            self._rl_algorithm.summarize_rollout(
                experience._replace(rollout_info=experience.rollout_info.rl))
