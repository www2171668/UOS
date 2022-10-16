
"""This is the configuration to train Procgen environment with PPO

Specifically, this configuration serves as an example that trains on
bossfight, but can be adapted to other Procgen games with slgiht
modification.

Note that the default mini batch size = 4096 currently consumes 23.5
GB of GPU memory. You will need at least an RTX 3090 for training.

Most of the hyper parameters are set with courtesy of OpenAI's Phasic
Policy Gradient implementation:
https://github.com/openai/phasic-policy-gradient

"""

import alf
import alf.examples.ppo_procgen.base_conf

# Environment Configuration
alf.config('create_environment', env_name='bossfight')
