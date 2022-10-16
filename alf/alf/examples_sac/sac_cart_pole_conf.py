

import alf
import alf.examples_sac.sac_conf
from alf.networks import ActorDistributionNetwork, QNetwork
from alf.utils.losses import element_wise_squared_loss
from alf.algorithms.sac_algorithm import SacAlgorithm

# environment config
alf.config(
    'create_environment', env_name="CartPole-v0", num_parallel_environments=8)

# algorithm config
alf.config('QNetwork', fc_layer_params=(100, ))
# note that for discrete action space we do not need the actor network as a
# discrete action can be sampled from the Q values.
alf.config(
    'SacAlgorithm',
    q_network_cls=QNetwork,
    actor_optimizer=alf.optimizers.Adam(lr=1e-3, name='actor'),
    critic_optimizer=alf.optimizers.Adam(lr=1e-3, name='critic'),
    alpha_optimizer=alf.optimizers.Adam(lr=1e-3, name='alpha'),
    target_update_tau=0.01)

alf.config(
    'OneStepTDLoss', td_error_loss_fn=element_wise_squared_loss, gamma=0.98)

# training config
alf.config(
    'TrainerConfig',
    initial_collect_steps=1000,
    mini_batch_length=2,
    mini_batch_size=64,
    unroll_length=1,
    num_updates_per_train_iter=1,
    num_iterations=10000,
    num_checkpoints=5,
    evaluate=False,
    eval_interval=100,
    debug_summaries=True,
    summary_interval=100,
    replay_buffer_length=100000)
