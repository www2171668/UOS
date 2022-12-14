

import functools
import torch

import alf
from alf.environments import suite_mario, alf_environment
from alf.environments import thread_environment, parallel_environment
import alf.nest as nest


class SuiteMarioTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_mario.is_available():
            self.skipTest('suite_mario is not available.')

    def tearDown(self):
        super().tearDown()
        self._env.close()

    def test_process_env(self):
        game = 'SuperMarioBros-Nes'

        self._env = suite_mario.load(
            game=game, state='Level1-1', wrap_with_process=True)
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self.assertEqual(torch.uint8, self._env.observation_spec().dtype)
        self.assertEqual((1, 84, 84), self._env.observation_spec().shape)

        actions = self._env.action_spec().sample()
        for _ in range(10):
            time_step = self._env.step(actions)

    def test_thread_env(self):
        game = 'SuperMarioBros-Nes'
        self._env = thread_environment.ThreadEnvironment(
            lambda: suite_mario.load(
                game=game, state='Level1-1', wrap_with_process=False))
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self.assertEqual(torch.uint8, self._env.observation_spec().dtype)
        self.assertEqual((1, 84, 84), self._env.observation_spec().shape)

        actions = self._env.action_spec().sample()
        for _ in range(10):
            time_step = self._env.step(actions)

    def test_parallel_env(self):
        game = 'SuperMarioBros-Nes'
        env_num = 8

        def ctor(game, env_id=None):
            return suite_mario.load(
                game=game, state='Level1-1', wrap_with_process=False)

        constructor = functools.partial(ctor, game)

        self._env = parallel_environment.ParallelAlfEnvironment(
            [constructor] * env_num)
        self.assertTrue(self._env.batched)
        self.assertEqual(self._env.batch_size, env_num)
        self.assertEqual(torch.uint8, self._env.observation_spec().dtype)
        self.assertEqual((1, 84, 84), self._env.observation_spec().shape)

        actions = self._env.action_spec().sample(outer_dims=(env_num, ))
        for _ in range(10):
            time_step = self._env.step(actions)


if __name__ == '__main__':
    alf.test.main()
