import alf
import torch
import copy
import numpy as np

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import TimeStep, AlgStep, Experience, make_experience, namedtuple
from .discriminator import Discriminator

from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.common import Periodically
from alf.utils.conditional_ops import conditional_update

from hidio.algorithm.random_goal import Random_Goal

SkillGeneratorState = namedtuple(
    "SkillGeneratorState", ["discriminator",
                            "rl", "skill", "rl_reward", "rl_discount", "steps"], default_value=())

SkillGeneratorInfo = namedtuple("SkillGeneratorInfo", ["skill", "reward", "switch_skill", "steps"], default_value=())

# % 技能生成器 = 上层sac + 判别器网络
@alf.configurable
class SkillGenerator(Algorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 skill_spec,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config: TrainerConfig = None,
                 num_steps_per_skill=3,
                 rl_algorithm_cls=SacAlgorithm,
                 rl_mini_batch_size=128,
                 rl_mini_batch_length=2,
                 rl_replay_buffer_length=20000,
                 disc_mini_batch_size=64,
                 disc_mini_batch_length=4,
                 disc_replay_buffer_length=20000,
                 gamma=0.99,
                 skill_noise=0.,
                 high_step_scale=5,
                 dynamic=False,
                 optimizer=None,
                 debug_summaries=False,
                 name="SkillGenerator"):
        # % 必要参数
        self._num_steps_per_skill = num_steps_per_skill
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._skill_spec = skill_spec
        self._gamma = gamma
        self._skill_noise = skill_noise
        self._high_step_scale = high_step_scale
        self._dynamic = dynamic

        # % 算法实例化 上层sac + disc
        rl, discriminator = self._create_subalgorithms(
            rl_algorithm_cls, debug_summaries, env, config,
            rl_mini_batch_size, rl_mini_batch_length, rl_replay_buffer_length,
            disc_mini_batch_size, disc_mini_batch_length, disc_replay_buffer_length)

        # % 定义state_spec
        train_state_spec = SkillGeneratorState(
            discriminator=discriminator.train_state_spec,  # for discriminator
            skill=self._skill_spec)  # inputs to lower-level
        rollout_state_spec = train_state_spec._replace(
            rl=rl.train_state_spec,  # higher-level policy rollout
            rl_reward=TensorSpec(()),  # higher-level policy replay
            rl_discount=TensorSpec(()),  # higher-level policy replay
            steps=TensorSpec((), dtype='int64'))
        predict_state_spec = train_state_spec._replace(
            discriminator=discriminator.predict_state_spec,  # for discriminator
            rl=rl.predict_state_spec,  # higher-level policy prediction
            steps=TensorSpec((), dtype='int64'))

        # % 继承
        super().__init__(
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            optimizer=optimizer,
            name=name)

        # % 必要参数
        self._random_goal = Random_Goal(skill_spec, env.batch_size) \
            if hasattr(env, 'batch_size') else Random_Goal(skill_spec, 1)  # train or play
        self._discriminator = discriminator
        self._rl = rl
        self._rl_train = Periodically(self._rl.train_from_replay_buffer,
                                      period=1,
                                      name="periodic_higher_level")

    # % 算法实例化
    def _create_rl_algorithm(self, rl_algorithm_cls, rl_config, env, debug_summaries):
        """Initial sac class. Pass env and rl_config for creating a replay buffer and metrics"""
        rl = rl_algorithm_cls(
            observation_spec=self._observation_spec,
            action_spec=self._skill_spec,  # high_action_spec
            config=rl_config,
            debug_summaries=debug_summaries)
        if env:
            rl.set_replay_buffer(env.batch_size, rl_config.replay_buffer_length,
                                 prioritized_sampling=False)
        return rl

    def _create_discriminator(self, disc_config, env, num_steps_per_skill, debug_summaries):
        """Initial disc class. Pass env and disc_config for creating a replay buffer and metrics"""
        discriminator = Discriminator(
            observation_spec=self._observation_spec,
            action_spec=self._action_spec,
            skill_spec=self._skill_spec,
            config=disc_config,
            num_steps_per_skill=num_steps_per_skill,
            debug_summaries=debug_summaries)
        if env:
            discriminator.set_replay_buffer(env.batch_size, disc_config.replay_buffer_length,
                                            prioritized_sampling=False)
        return discriminator

    def _create_subalgorithms(self, rl_algorithm_cls, debug_summaries, env, config,
                              rl_mini_batch_size, rl_mini_batch_length, rl_replay_buffer_length,
                              disc_mini_batch_size, disc_mini_batch_length, disc_replay_buffer_length):
        # * 设置参数
        rl_config = copy.deepcopy(config)
        disc_config = copy.deepcopy(config)
        if config is not None:
            rl_config.mini_batch_size = rl_mini_batch_size
            rl_config.mini_batch_length = rl_mini_batch_length
            rl_config.replay_buffer_length = rl_replay_buffer_length
            rl_config.initial_collect_steps = config.initial_collect_steps * self._high_step_scale

            disc_config.mini_batch_size = disc_mini_batch_size
            disc_config.mini_batch_length = disc_mini_batch_length
            disc_config.replay_buffer_length = disc_replay_buffer_length
            disc_config.initial_collect_steps = config.initial_collect_steps

        rl = self._create_rl_algorithm(rl_algorithm_cls, rl_config, env, debug_summaries)
        discriminator = self._create_discriminator(disc_config, env, self._num_steps_per_skill, debug_summaries)
        return rl, discriminator

    # % 属性方法
    @property
    def skill_spec(self):
        return self._skill_spec

    @property
    def num_steps_per_skill(self):
        return self._num_steps_per_skill

    # % 切换技能判定
    def _should_switch_skills(self, time_step: TimeStep, state: SkillGeneratorState):
        if not self._dynamic:  # * 复合调度
            should_switch_skills = ((state.steps % self._num_steps_per_skill) == 0)
            disc_switch = rl_switch = should_switch_skills | time_step.is_first() | time_step.is_last()
        else:  # * 动态调度
            dynamic_switch_skills = alf.nest.map_structure(lambda repeats: repeats<=1, state.rl.repeats)
            disc_switch = dynamic_switch_skills | time_step.is_first() | time_step.is_last()
            rl_switch = torch.tensor([True])
        return disc_switch, rl_switch

    # % skill play
    def gen_random_skill(self, time_step: TimeStep, state: SkillGeneratorState):
        switch_skill, switch_rl = self._should_switch_skills(time_step, state)
        state = conditional_update(target=state, cond=switch_rl, func=self._clear_step, state=state)

        discriminator_step = self._discriminator_predict_step(time_step, state, switch_skill)
        random_skill = self._random_goal._update_skill(state, time_step.step_type)
        state = state._replace(skill=random_skill)

        new_state = state._replace(discriminator=discriminator_step.state, steps=state.steps + 1)
        info = SkillGeneratorInfo(switch_skill=switch_skill)
        return AlgStep(output=new_state.skill, state=new_state, info=info)

    def _clear_step(self, state):
        state = state._replace(steps=torch.zeros_like(state.steps))
        return state

    # % sac基础 + disc基础
    def predict_step(self, time_step: TimeStep, state: SkillGeneratorState):
        """every ``self._num_steps_per_skill`` calls ``self._rl`` to generate new skills."""
        switch_skill, switch_rl = self._should_switch_skills(time_step, state)

        discriminator_step = self._discriminator_predict_step(time_step, state, switch_skill)
        new_state = conditional_update(
            target=state,
            cond=switch_rl,
            func=self._rl_predict_step,
            time_step=time_step,
            state=state)

        new_state = new_state._replace(discriminator=discriminator_step.state, steps=new_state.steps + 1)
        info = SkillGeneratorInfo(switch_skill=switch_skill)
        return AlgStep(output=new_state.skill, state=new_state, info=info)

    def rollout_step(self, time_step: TimeStep, state: SkillGeneratorState):
        """
           1. every ``self._num_steps_per_skill`` it calls ``self._rl`` to generate new skills.
              Writes rl's ``time_step`` to a replay buffer when new skills are generated.
           2. call discriminator's ``rollout_step()``
              Writes discriminator's ``time_step`` to a replay buffer
        """
        switch_skill, switch_rl = self._should_switch_skills(time_step, state)

        # % discriminator
        discriminator_step = self._discriminator_rollout_step(time_step, state, switch_skill)

        # % high_sac
        state = state._replace(
            rl_reward=state.rl_reward + state.rl_discount * time_step.reward,
            rl_discount=state.rl_discount * self._gamma * time_step.discount)
        new_state = conditional_update(
            target=state,
            cond=switch_rl,
            func=self._rl_rollout_step,
            time_step=time_step,
            state=state)

        new_state = new_state._replace(
            discriminator=discriminator_step.state,
            steps=new_state.steps + 1)
        info = SkillGeneratorInfo(
            skill=new_state.skill,
            steps=new_state.steps,
            switch_skill=switch_skill)
        return AlgStep(output=new_state.skill, state=new_state, info=info)

    def train_step(self, inputs: TimeStep, state: SkillGeneratorState, rollout_info: SkillGeneratorInfo):
        """
        1. take the skill generated during ``rollout_step`` and output it as the
           skill for the current time step.
        2. generate intrinsic rewards using the discriminator (fixed), for training
           the skill-conditioned policy.
        """
        discriminator_step = self._discriminator_train_step(inputs, state, rollout_info)

        # % 轨迹泛化
        if self._skill_noise:
            # noise = torch.Tensor(1).uniform_(1 - self._skill_noise, 1 + self._skill_noise)
            noise = torch.Tensor(rollout_info.skill.shape).uniform_(1 - self._skill_noise, 1 + self._skill_noise)
            noise_skill = alf.nest.map_structure(lambda skill: torch.multiply(skill, noise), rollout_info.skill)
            rollout_info = rollout_info._replace(skill=noise_skill)

        new_state = state._replace(discriminator=discriminator_step.state, skill=rollout_info.skill)
        info = SkillGeneratorInfo(reward=discriminator_step.info)
        return AlgStep(output=new_state.skill, state=new_state, info=info)

    def _trainable_attributes_to_ignore(self):
        """These paras will train themselves, so let the parent algorithm ignore them"""
        return ["_rl", "_discriminator"]

    def after_train_iter(self, experience, train_info):
        # % 训练high_sac
        with alf.summary.scope(self.name + "_rl"):
            self._rl.train_from_replay_buffer()

        # % 训练discriminator
        with alf.summary.scope(self._discriminator.name):
            self._discriminator.train_from_replay_buffer()

    # % —————— sac基础 ————————
    def _rl_predict_step(self, time_step, state):
        rl_step = self._rl.predict_step(time_step, state.rl)
        new_state = state._replace(rl=rl_step.state,
                                   skill=rl_step.output,
                                   steps=torch.zeros_like(state.steps))
        return new_state

    def _rl_rollout_step(self, time_step, state):
        """Suppose that during an episode we have :math:`H` segments where each segment
        contains :math:`K` steps. Then the objective for the higher-level policy is:

        .. math::
            \begin{array}{ll}
                &\sum_{h=0}^{H-1}(\gamma^K)^h\sum_{t=0}^{K-1}\gamma^t r(s_{t+hK},a_{t+hK})\\
                =&\sum_{h=0}^{H-1}\beta^h R_h\\
            \end{array}

        :math:`\beta=\gamma^K` is the discount per higher-level time step and
        :math:`R_h=\sum_{t=0}^{K-1}\gamma^t r(s_{t+hK},a_{t+hK})` is reward per higher-level time step."""
        rl_time_step = time_step._replace(
            reward=state.rl_reward,
            discount=state.rl_discount,
            prev_action=state.skill)
        rl_step = self._rl.rollout_step(rl_time_step, state.rl)

        if False:
            if self._skill_spec.is_discrete:
                rl_step = rl_step._replace(output=torch.tensor([0]))
            else:
                rang_skill = torch.Tensor(rl_step.output.shape).uniform_(-0.1, 0.1)
                rl_step = rl_step._replace(output=torch.tensor(rang_skill))

        new_state = state._replace(discriminator=state.discriminator,
                                   rl=rl_step.state,
                                   skill=rl_step.output,  # high_sac -> skill
                                   rl_reward=torch.zeros_like(state.rl_reward),  # [0...]
                                   rl_discount=torch.ones_like(state.rl_discount),  # [1...]
                                   steps=torch.zeros_like(state.steps))

        # * 存储经验在high replay buffer
        untransformed_time_step = rl_time_step._replace(
            observation=rl_time_step.untransformed.observation)  # rl_time_step.observation has been transformed
        exp = make_experience(untransformed_time_step.cpu(), rl_step, state.rl)
        self._rl.observe_for_replay(exp)
        return new_state

    # % —————— 判别器基础 ————————
    def _discriminator_predict_step(self, time_step, state: SkillGeneratorState, switch_skill):
        observation = [time_step.observation, switch_skill]
        time_step = time_step._replace(observation=observation)
        discriminator_step = self._discriminator.predict_step(time_step, state.discriminator)
        return discriminator_step

    def _discriminator_rollout_step(self, time_step, state: SkillGeneratorState, switch_skill):
        observation = [time_step.observation, state.skill, switch_skill,
                       state.steps % self._num_steps_per_skill + 1]  # [s, u, switch, steps]
        time_step = time_step._replace(observation=observation)
        discriminator_step = self._discriminator.rollout_step(time_step, state.discriminator)

        time_step = time_step._replace(reward=time_step.prev_action.new_zeros((1,)))
        disc_exp = make_experience(time_step.cpu(), discriminator_step, state.discriminator)
        self._discriminator.observe_for_replay(disc_exp)

        return discriminator_step

    def _discriminator_train_step(self, inputs: TimeStep, state: SkillGeneratorState, rollout_info):
        """calc intrinsic_reward.  discriminator_step.info=intrinsic_reward

        Issue: this steps will be inaccurate if FINAL step comes before num_steps_per_skill"""
        observation = [inputs.observation, state.skill, rollout_info.switch_skill, rollout_info.steps]  # [s, u, switch, steps]
        disc_inputs = inputs._replace(observation=observation)
        discriminator_step = self._discriminator.train_step(disc_inputs, state.discriminator, rollout_info, trainable=False)
        return discriminator_step
