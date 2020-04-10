"""
With the default input value, which is the most rewarding method
over 100 runs with different initial random distributions?
"""
import numpy as np

from tqdm import tqdm

from mab.game import Game


K = 10
num_trials_per_strategy = 100
strategies = [
    "naive",
    "best reward",
    "least explored",
    "upper confidence",
    "gradient"
]

average_rewards_per_method = np.zeros(
    [len(strategies), K], dtype=np.float
)


for strat_n, strat in enumerate(strategies):
    print(f"Strategy employed: {strat}")
    cumulative_reward_per_arm_per_strategy = np.zeros([K], dtype=np.float)

    for i in tqdm(range(num_trials_per_strategy)):
        np.random.seed(i)
        game = Game(1000)

        _, _, reward_per_arm, _ = game.play(
            initial_t_explorations=100, exploration_strategy=strat
        )
        cumulative_reward_per_arm_per_strategy += reward_per_arm

    average_rewards_per_method[strat_n, :] = cumulative_reward_per_arm_per_strategy / num_trials_per_strategy

total_rewards_per_strategy = np.sum(average_rewards_per_method, axis=1)
best_strategy_index = np.argmax(total_rewards_per_strategy)
print(f"The strategy that provided the highest total reward on the average of {num_trials_per_strategy} cases is:")
print(total_rewards_per_strategy)
print(strategies[best_strategy_index])
