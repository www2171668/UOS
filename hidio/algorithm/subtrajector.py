""""""
import gin
import time
import torch

import alf
from alf.utils.conditional_ops import conditional_update
from alf.data_structures import TimeStep, Experience, namedtuple, AlgStep

SubTrajectory = namedtuple('SubTrajectory', ["observation", "prev_action"], default_value=())

DiscriminatorState = namedtuple("DiscriminatorState", ["first_observation", "untrans_observation", "subtrajectory"],
                                default_value=())

# % 对low_sac
def make_low_rl_observation(subtrajectory, updated_first_observation,
                            skill, steps, switch_skill, num_steps_per_skill):
    """Given skill, makes the skill-conditioned observation for lower-level policy.

       Both observation and action are a stacking of recent states, with the most recent one appearing at index=0.
       X: first observation of a skill
       _: zero
       """
    update_lines = torch.arange(updated_first_observation.shape[0]).long()
    subtrajectory.observation[update_lines, steps - 1] = updated_first_observation

    def _zero(subtrajectory):
        subtrajectory.observation[:, 1:, ...] = 0.
        subtrajectory.prev_action.fill_(0.)
        return subtrajectory

    subtrajectory = conditional_update(
        target=subtrajectory,
        cond=switch_skill,
        func=_zero,
        subtrajectory=subtrajectory)
    subtrajectory = alf.nest.map_structure(lambda traj: traj.reshape(traj.shape[0], -1), subtrajectory)
    subtrajectory = alf.nest.flatten(subtrajectory)

    low_rl_observation = (subtrajectory + [num_steps_per_skill - steps, skill])
    return low_rl_observation

# % first_observation
def update_state_if_necessary(switch_skill, obs, first_obs):
    return torch.where(torch.unsqueeze(switch_skill, dim=-1), obs, first_obs)

def clear_subtrajectory_if_necessary(switch_skill, subtrajectory):
    def _clear(subtrajectory):
        zeros = torch.zeros_like(subtrajectory)
        return torch.where(switch_skill.reshape(-1, 1, 1).expand(-1, *zeros.shape[1:]),
                           zeros, subtrajectory)

    return alf.nest.map_structure(_clear, subtrajectory)

def update_subtrajectory(time_step: TimeStep, state: DiscriminatorState):
    def _update(subtrajectory, subtrajectory_unit):
        new_subtrajectory = torch.roll(subtrajectory, shifts=1, dims=1)
        new_subtrajectory[:, 0, ...] = subtrajectory_unit
        subtrajectory.copy_(new_subtrajectory)
        return new_subtrajectory

    subtrajectory = SubTrajectory(
        observation=state.subtrajectory.observation,
        prev_action=state.subtrajectory.prev_action)
    subtrajectory_unit = SubTrajectory(  # (s,a-)
        observation=time_step.observation,
        prev_action=time_step.prev_action)

    return alf.nest.map_structure(_update, subtrajectory, subtrajectory_unit)
