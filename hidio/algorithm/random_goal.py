""""""

import gin
import alf
import numpy as np
import torch

from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple, AlgStep, StepType
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
import alf.utils.common as common

class Random_Goal:
    def __init__(self, skill_spec, batch=1):
        self.Episode = -1
        self.zip_line = []
        for i in np.linspace(-0.8, 0.8, 5):
            for j in np.linspace(-0.8, 0.8, 5):
                self.zip_line.append((i, j))  # huang

        if skill_spec.is_discrete:
            assert isinstance(skill_spec, BoundedTensorSpec)
            skill_dim = skill_spec.maximum - skill_spec.minimum + 1
        else:
            assert len(skill_spec.shape) == 1, "Only 1D skill vector is supported"
            skill_dim = skill_spec.shape[0]

        self._skill_spec = skill_spec
        self._skill_dim = skill_dim
        self._batch = batch

    # % continuous
    def _cont_order_random(self):
        skills = torch.tensor([[-1 + self.Episode / 20]])
        return skills

    def _cont_order_random_2(self):
        skills = torch.tensor([[-1 + self.Episode / 20, -1 + self.Episode / 20]])
        return skills

    def _cont_skill_random(self):
        skills = torch.rand([1, self._skill_dim])  # [[*,*,*,*]]
        skills = 2 * skills - 1
        return skills

    def _cont_skill_gauss(self):
        skills = torch.normal(0, 1, size=(1, self._skill_dim))
        return skills

    def _cont_skill_line(self):
        skills = self.zip_line[self.Episode]
        skills = torch.Tensor([skills])
        return skills

    # % discrete
    def _dis_order_skill(self):
        skills = torch.tensor([self.Episode])
        return skills

    def _dis_random_skill(self):
        skills = torch.randint(high=self._skill_dim, size=(self._batch,), dtype=torch.int64)
        return skills

    def _one_hot_skill(self):
        skills = torch.randint(high=self._skill_dim, size=(self._batch,), dtype=torch.int64)
        skills_onehot = torch.nn.functional.one_hot(skills, self._skill_dim).to(torch.float32)
        return skills_onehot

    def _update_skill(self, state, step_type):
        """Update the skill if the episode just beginned; otherwise keep using the skill in ``state``."""
        new_skill_mask = torch.unsqueeze((step_type == StepType.FIRST), dim=-1)
        if new_skill_mask:
            self.Episode += 1

        if self._skill_spec.is_discrete:
            assert self.Episode < self._skill_dim, "All skill done"
            generated_skill = self._dis_order_skill()
            new_skill = torch.where(new_skill_mask, generated_skill, state.skill).squeeze(1)
        else:
            generated_skill = self._cont_skill_random()
            new_skill = torch.where(new_skill_mask, generated_skill, state.skill)

        return new_skill
