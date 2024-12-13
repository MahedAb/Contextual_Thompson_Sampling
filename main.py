import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import argparse


class ThompsonSampling:
    def __init__(self, dim=10, reward_simulator=None, prior_mean=0, reg_lambda=1, exploit_alpha=1):
        """
        Initialize the Thompson Sampling algorithm.
        :param: reward_simulator: the reward simulator class for generating the rewards
        :param prior_mean: set to 0
        :param reg_lambda: set to 1/q
        :param exploit_alpha: smaller alpha encourages exploitation as described in the paper
        """
        self.dim = dim
        self.reward_sim = reward_simulator
        self.prior_mean = np.ones(dim) * prior_mean
        self.prior_std = np.ones(dim) * np.sqrt(1 / reg_lambda) * exploit_alpha
        self.set_action = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.entire_data = []  # Initialize D as an empty dataset

    def run(self, context_batch):
        """
        Run the Thompson Sampling algorithm for a given context batch
        """
        n = context_batch.shape[0]
        action_batch = []
        reward_batch = []

        for i in range(n):
            # Step 1: Receive context xt
            context = context_batch[i, :]

            # Step 2: Draw w_t according to P(w|D)
            sampled_weights = np.random.multivariate_normal(self.prior_mean, np.diag(self.prior_std))

            # Step 3: Select best action given sampled weights
            action = self.select_action(context, sampled_weights)
            action_batch.append(action)

            # Step 4: Observe reward
            r = self.reward_sim.generate(context, action)  # Replace with real-world interaction
            reward_batch.append(r)

            # Step 5: Update the database D = D ∪ {(x, a, r)}
            self.entire_data.append((context, action, r))

        # Update the model
        self.update_model(context_batch, action_batch, reward_batch)

        return reward_batch, action_batch

    def select_action(self, context, sampled_weights):
        """
        Select the best action based on sampled weights
        """
        # Compute expected reward for each possible action
        expected_rewards = []
        # Maximise the reward for the sampled weights
        for action in self.set_action:
            x = np.concatenate([action, context])
            expected_rewards.append((1 / (1 + np.exp(-np.dot(x, sampled_weights)))))
        return self.set_action[np.argmax(expected_rewards)]

    def logistic_loss(self, w, x, y, m, q):

        """
        Compute the regularized logistic regression loss
        :param w: Weight vector
        :param x: Input features (action and context)
        :param y: Labels (rewards)
        :param m: Prior mean
        :param q: Prior std
        :return: Loss value
        """
        # Regularization term
        reg_term = 0.5 * np.sum(q * (w - m) ** 2)

        # Logistic loss term
        logistic_term = np.sum(np.log(1 + np.exp(-y * np.dot(x, w))))

        return reg_term + logistic_term

    def update_model(self, contexts, actions, rewards):

        x = np.concatenate([actions, contexts], axis=1)
        w0 = np.random.multivariate_normal(self.prior_mean, np.diag(self.prior_std))
        result = minimize(
            lambda w: self.logistic_loss(w, x, np.array(rewards), self.prior_mean, self.prior_std),
            w0,
            method='BFGS'
        )
        w_opt = result.x
        self.prior_mean = w_opt

        # Compute Hessian (Fisher Information Matrix)
        n = contexts.shape[0]
        p = np.zeros(n)
        for j in range(n):
            p[j] = 1 / (1 + np.exp(-np.dot(x[j], w_opt)))
        for i in range(self.dim):
            temp = 0
            for j in range(n):
                temp += (x[j][i] ** 2) * p[j] * (1 - p[j])
            self.prior_std[i] += temp


class Reward:
    def __init__(self, dim):
        self.w = np.random.uniform(-1, 1, size=dim)

    def generate(self, context, action):
        """
        Simulate a reward for action and context
        """
        x = np.concatenate([action, context])
        rewards = (1 / (1 + np.exp(-np.dot(x, self.w))) > 0.5).astype(int)
        return rewards


def create_context(n_batches, dim):
    """
    Simple context creator
    """

    return np.random.normal(0, 1, size=(n_batches, dim))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=100, help="number of batches (runs)")
    parser.add_argument('--n', type=int, default=10, help="batch size")
    args = parser.parse_args()

    T = args.T
    n = args.n
    d = 10  # dimension of X: action + context (we assume the first two are actions)

    reward_sim = Reward(d)
    TS = ThompsonSampling(dim=d, reward_simulator=reward_sim, exploit_alpha=1)

    average_rewards = []
    for i in range(T):
        contexts = create_context(n, d - 2)
        reward, actions = TS.run(contexts)
        average_rewards.append(np.mean(reward))
        print(actions)

    plt.plot(range(T), average_rewards)
    plt.xlabel('Steps (T)')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time')
    plt.savefig('output/average_reward_plot.png')
