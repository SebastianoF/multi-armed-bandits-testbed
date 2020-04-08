import numpy as np


def epsilon_greedy(
        player, initial_t_explorations: int, initial_k: int = None, epsilon: float =0.1, alpha_override=None,
        exploration_strategy="random"
):
    """
    Epsilon greedy algorithm and variant with multiple strategies. Prototype not optimized for speed.
    Source: Reinforcement Learning, an introduction. Sutton 2018
    """
    def normalize(v):
        if np.sum(v) > 0:
            return v / np.sum(v)
        else:
            return np.zeros_like(v) + 1 / len(v)

    def get_another_k_random():
        return np.random.choice(list(set(range(player.mab.K)) - {k}))

    def get_another_k_best_reward():
        # also Scrooge greedy
        weights = np.copy(rewards_per_arm)
        weights[k] = 0
        return np.random.choice(np.arange(player.mab.K), p=normalize(weights))

    def get_another_k_least_explored():
        weights = np.max(pulls_per_arm) - pulls_per_arm
        weights[k] = 0
        return np.random.choice(np.arange(player.mab.K), p=normalize(weights))

    def get_another_k_upper_confidence_bound():
        return np.argmax(means_hat + [0.3 * np.log(t + 0.000001) / (n + 1) for n in pulls_per_arm])

    def get_another_k_gradient():
        weights = np.exp(H[t, :]) / np.sum(np.exp(H[t, :]))
        # update matrix H for the next step.
        avg_reward_per_arm = np.array([r / p if p else 0 for r, p in zip(rewards_per_arm, pulls_per_arm)])
        H[t + 1, :] = H[t, :] - 0.3 * (rewards_per_arm - avg_reward_per_arm) * weights
        H[t + 1, k] = H[t, k] + 0.3 * (rewards_per_arm[k] - avg_reward_per_arm[k]) * (1 - weights[k])
        return np.random.choice(np.arange(player.mab.K), p=normalize(weights))

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

    if not initial_k:
        k = np.random.choice(player.mab.K)
    else:
        k = initial_k

    means_hat = np.zeros(player.mab.K, dtype=np.float)
    stds_hat = np.zeros(player.mab.K, dtype=np.float)
    pulls_per_arm = np.zeros(player.mab.K, dtype=np.int)  # N
    rewards_per_arm = np.zeros(player.mab.K, dtype=np.float)  # R

    if exploration_strategy == "gradient":
        H = np.zeros([player.T + 1, player.mab.K], dtype=np.float)

    # initial step:
    q = player.mab.draw_from_arm(k)

    player.q[k, 0] = q

    rewards_per_arm[k] += np.max([0, q])
    pulls_per_arm[k] += 1

    for t in range(1, player.T):
        # chose the next arm:
        if t < initial_t_explorations:
            # pure explorative initial phase
            k = np.random.choice(player.mab.K)
        else:
            if np.random.rand() < epsilon:
                # explore a new arm
                k = get_another_k()
            else:
                # exploit more rewarding arm
                k = np.argmax(rewards_per_arm)

        # pull it (we do not lose money in this model, if the arm gives negative reward.)
        q = player.mab.draw_from_arm(k)

        # update values
        player.q[k, t] = np.max([0, q])
        rewards_per_arm[k] += np.max([0, q])
        pulls_per_arm[k] += 1

        # cumulative mean and standard deviation, non constant alpha for stationary problems:
        if alpha_override is None:
            alpha = (1 / pulls_per_arm[k])
        else:
            alpha = alpha_override

        means_hat[k] = means_hat[k] + alpha * (q - means_hat[k])
        stds_hat[k] = stds_hat[k] + alpha * ((q - means_hat[k]) ** 2 - stds_hat[k])

    player.total_reward = np.sum(rewards_per_arm)

    return means_hat, stds_hat, rewards_per_arm, pulls_per_arm
