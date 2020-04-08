import numpy as np


def epsilon_greedy(
        player, initial_t_explorations: int, initial_k: int = None, epsilon: float =0.1, alpha=None,
        exploration_strategy="random"
):
    """
    Epsilon greedy algorithm and variant with multiple strategies. Prototype not optimized for speed.
    Source: Reinforcement Learning, an introduction. Sutton 2018
    """
    def get_another_k_random():
        return np.random.choice(list(set(range(player.mab.K)) - {k}))

    def get_another_k_best_reward():
        # also Scrooge greedy
        weights = reward_per_arm
        weights[k] = 0
        return np.random.choice(np.arange(player.mab.K), weights)

    def get_another_k_least_explored():
        weights = pulls_per_arm - np.max(pulls_per_arm)
        weights[k] = 0
        return np.random.choice(np.arange(player.mab.K), weights)

    def get_another_k_upper_confidence_bound():
        return np.argmax(mus_hat + [0.3 * np.ln(t) / (n + 1) for n in pulls_per_arm])

    def get_another_k_gradient():
        weights = np.exp(H[t, :]) / np.sum(np.exp(H[t, :]))
        # update matrxi H for the next step.
        avg_reward_per_arm = np.array([r / p if p else 0 for r, p in zip(reward_per_arm, pulls_per_arm)])
        H[t + 1, :] = H[t + 1, :] - 0.3 * (reward_per_arm - avg_reward_per_arm) * weights
        H[t + 1, k] = H[t + 1, k] + 0.3 * (reward_per_arm[k] - avg_reward_per_arm[k]) * (1 - weights[k])
        return np.random.choice(np.arange(player.mab.K), weights)

    get_another_k_map = {
        "random": get_another_k_random,
        "best reward": get_another_k_best_reward,
        "least explored": get_another_k_least_explored,
        "upper confidence": get_another_k_upper_confidence_bound,
        "gradient": get_another_k_gradient
    }

    try:
        get_another_k = get_another_k_map[exploration_strategy]
    except ValueError:
        raise ValueError(
            f"input 'exploration strategy must be in {get_another_k_map.keys()}'. "
            f"Got {exploration_strategy}."
        )

    player.reset_parameters()

    if player.seed is not None:
        np.random.seed(player.seed)
    if not initial_k:
        k = np.random.choice(player.mab.K + 1)
    else:
        k = initial_k

    mus_hat = np.zeros(player.mab.K, dtype=np.float)
    stds_hat = np.zeros(player.mab.K, dtype=np.float)
    pulls_per_arm = np.zeros(player.mab.K, dtype=np.int)  # N
    reward_per_arm = np.zeros(player.mab.K, dtype=np.int)  # R

    if exploration_strategy == "gradient":
        H = np.zeros([player.T, player.mab.K + 1], dtype=np.float)

    for t in range(player.T):
        q = player.mab.draw_from_arm(k)
        player.q[k, t] = q
        # we do not lose money in this model, if the arm gives negative reward.
        reward_per_arm[k] += np.min(0, q)
        pulls_per_arm[k] += 1

        # cumulative mean and standard deviation, non constant alpha for stationary problems:
        if alpha is None:
            alpha = (1 / pulls_per_arm[k])
        mus_hat[k] += alpha * (q - mus_hat[k])
        stds_hat[k] += alpha * ((q - mus_hat[k]) ** 2 - stds_hat[k])

        if t < initial_t_explorations:
            # pure explorative initial phase
            k = get_another_k()
        else:
            if np.random.rand() < epsilon:
                # explore a new arm
                k = get_another_k()
            else:
                # exploit more rewarding arm
                k = np.argmax(reward_per_arm)

    player.total_reward = np.sum(reward_per_arm)

    return mus_hat, stds_hat, reward_per_arm, pulls_per_arm
