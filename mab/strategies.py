import numpy as np


def epsilon_greedy(
        player, initial_t_explorations: int, initial_k: int = None, epsilon: float =0.1, alpha=None,
        exploration_strategy="random"
):
    """
    Epsilon greedy algorithm and variant with multiple strategies. Prototype not optimized for speed.
    Source: Reinforcement Learning, an introduction. Sutton 2018
    """
    def get_another_k():
        if exploration_strategy == "random":
            return np.random.choice(list(set(range(player.mab.K)) - {k}))
        elif exploration_strategy == "best_reward":  # also Scrooge greedy
            weights =  reward_per_arm
            weights[k] = 0
            return np.random.choice(np.arange(player.mab.K), weights)
        elif exploration_strategy == "least explored":  # also optimistic
            weights = pulls_per_arm - np.max(pulls_per_arm)
            weights[k] = 0
            return np.random.choice(np.arange(player.mab.K), weights)
        elif exploration_strategy == "upper_confidence_bound":  # further exploration selection
            return np.argmax(mus_hat + [0.3 * np.ln(t) / (n + 1) for n in pulls_per_arm])
        else:
            raise ValueError(f"Exploration strategy '{exploration_strategy}' not allowed.")

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


def bayesian(player):
    """A modern Bayesian look at the multi-armed bandit. S Scott 2010"""
    raise NotImplementedError()
