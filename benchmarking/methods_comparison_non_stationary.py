import numpy as np

from tqdm import tqdm

from mab.game import Game


K = 10
timepoints = 500
num_trials_per_strategy = 100
initial_t_explorations = 20
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


np.random.seed(40)

means = np.zeros([timepoints, K], dtype=np.float)
stds = np.zeros([timepoints, K], dtype=np.float)

means[0, :] = np.array([3 * np.sin((np.pi / 4) * x) for x in range(K)])
stds[0, :] = np.random.uniform(1, 3, size=K)

for row in range(1, timepoints):
    means[row, :] = np.array([3 * np.sin((np.pi / 4) * x + 0.05 * row + 3) for x in range(K)])
    stds[row, :] = stds[row - 1, :]  # + 0.1 * np.random.choice([-1, 1], size=[1, K], p=(.2, .8))


for strat_n, strat in enumerate(strategies):
    print(f"Strategy employed: {strat}")
    cumulative_reward_per_arm_per_strategy = np.zeros([K], dtype=np.float)

    for i in tqdm(range(num_trials_per_strategy)):
        np.random.seed(i)
        game = Game(timepoints, means, stds)

        _, _, reward_per_arm, _ = game.play(
            initial_t_explorations=100, exploration_strategy=strat, epsilon=0.2, adjust_alpha=False
        )
        cumulative_reward_per_arm_per_strategy += reward_per_arm

    average_rewards_per_method[strat_n, :] = cumulative_reward_per_arm_per_strategy / num_trials_per_strategy

total_rewards_per_strategy = np.sum(average_rewards_per_method, axis=1)
best_strategy_index = np.argmax(total_rewards_per_strategy)
print(f"The strategy that provided the highest total reward on the average of {num_trials_per_strategy} cases is:")
print(total_rewards_per_strategy)
print(strategies[best_strategy_index])
