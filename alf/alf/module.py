
"""Patch torch.nn.Module for better performance.

``torch.nn.Module.__getattr__`` is freqently used by all class derived from
``nn.Module``. It can inrtroduce too much unnecessary overhead. So we patch
``nn.Module`` class to remove it.
"""

import torch
from torch.nn import Module, Parameter

del Module.__getattr__

_old_Module__setattr__ = torch.nn.Module.__setattr__


def _new_Module__setattr__(self, name, value):
    _old_Module__setattr__(self, name, value)
    object.__setattr__(self, name, value)


Module.__setattr__ = _new_Module__setattr__

old_register_parameter = torch.nn.Module.register_parameter


def _new_register_parameter(self, name, param):
    old_register_parameter(self, name, param)
    object.__setattr__(self, name, param)


Module.register_parameter = _new_register_parameter

old_register_buffer = torch.nn.Module.register_buffer


def _new_register_buffer(self, name, param):
    old_register_buffer(self, name, param)
    object.__setattr__(self, name, param)


Module.register_buffer = _new_register_buffer
