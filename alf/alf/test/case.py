
"""Simple wrapper over unittest.TestCase to provide extra functionality."""

import torch
import unittest
import alf
from alf.utils import common


class TestCase(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.addTypeEqualityFunc(torch.Tensor, 'assertTensorEqual')

    def setUp(self):
        common.set_random_seed(1)
        alf.summary.reset_global_counter()

    def assertTensorEqual(self, t1, t2, msg=None):
        self.assertIsInstance(t1, torch.Tensor,
                              'First argument is not a Tensor')
        self.assertIsInstance(t2, torch.Tensor,
                              'Second argument is not a Tensor')

        if not torch.all(t1.cpu() == t2.cpu()):
            standardMsg = '%s != %s' % (t1, t2)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertTensorClose(self, t1, t2, epsilon=1e-6, msg=None):
        self.assertIsInstance(t1, torch.Tensor,
                              'First argument is not a Tensor')
        self.assertIsInstance(t2, torch.Tensor,
                              'Second argument is not a Tensor')
        diff = torch.max(torch.abs(t1 - t2))
        if not (diff <= epsilon):
            standardMsg = '%s is not close to %s. diff=%s' % (t1, t2, diff)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertTensorNotClose(self, t1, t2, epsilon=1e-6, msg=None):
        self.assertIsInstance(t1, torch.Tensor,
                              'First argument is not a Tensor')
        self.assertIsInstance(t2, torch.Tensor,
                              'Second argument is not a Tensor')
        if torch.max(torch.abs(t1 - t2)) < epsilon:
            standardMsg = '%s is actually close to %s' % (t1, t2)
            self.fail(self._formatMessage(msg, standardMsg))
