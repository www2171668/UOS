

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.trac_algorithm import TracAlgorithm
from alf.algorithms.data_transformer import RewardScaling

# environment config
alf.config(
    'create_environment', env_name="CartPole-v0", num_parallel_environments=8)

# reward scaling
alf.config('TrainerConfig', data_transformer_ctor=RewardScaling)
alf.config('RewardScaling', scale=0.01)

# algorithm config
alf.config('ActorDistributionNetwork', fc_layer_params=(100, ))
alf.config('ValueNetwork', fc_layer_params=(100, ))
alf.config(
    'ActorCriticAlgorithm',
    optimizer=alf.optimizers.Adam(lr=1e-3, gradient_clipping=10.0))
alf.config(
    'ActorCriticLoss',
    entropy_regularization=1e-4,
    gamma=0.98,
    use_gae=True,
    use_td_lambda_return=True)

# training config
alf.config(
    'TrainerConfig',
    unroll_length=10,
    algorithm_ctor=TracAlgorithm,
    num_iterations=2500,
    num_checkpoints=5,
    evaluate=True,
    eval_interval=500,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    summary_interval=5,
    epsilon_greedy=0.1)
