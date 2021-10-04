import os
import numpy as np
import matplotlib.pyplot as plt

class Boltzmann_exploration:
    def __init__(self, T, total_steps, k = 10) -> None:
        self.T = T # set Boltzmann's temperature constant
        self.k = k # number of bandits/actions
        self.actions = np.arange(k) # enumerate all the actions
        self.Q = np.zeros(k) # value estimates for each action
        self.k_count = np.ones(k) # initialize the count for each action [initialize to 1 to prevent zero division]
        self.t = 1 # time step count
        self.total_steps = total_steps
        self.expected_reward = np.zeros(total_steps)
        self.action_value = np.random.normal(1, 1, k) # sample action value for each action
        self.actual_rewards = np.zeros(total_steps) # this reward is sampled from N(action_value, 1) for each timestep

    def get_action(self) -> int:
        action_prob = np.exp(self.Q * self.T) / np.sum(np.exp(self.Q * self.T), axis=0)
        action = np.random.choice(self.actions, p=action_prob)
        return action_prob, action

    def get_reward_and_estimate_value(self) -> None:
        pi, action = self.get_action()
        reward = np.random.normal(self.action_value[action], 1)
        self.actual_rewards[self.t - 1] = reward

        # calculate expected reward
        expected_reward = pi * reward
        self.expected_reward[self.t - 1] = np.sum(expected_reward)
        # print(f'pi: {pi}, reward: {reward}, expected_reward: {expected_reward}')
        # print(f'self.expected_reward: {self.expected_reward}')

        # increment total and action specific counts
        # This is done after the rewards have been calculated.
        self.t += 1
        self.k_count[action] += 1

        # calculate the average (expected) reward for selected action
        # TODO: should the reward here be the just sampled reward from N(action_value, 1)???
        self.Q[action] = self.Q[action] + (reward - self.Q[action]) / self.k_count[action]


    def run_experiments(self):
        for i in range(self.total_steps):
            self.get_reward_and_estimate_value()

    def __call__(self):
        self.run_experiments()


if __name__ == '__main__':
    temperatures = [1, 3, 10, 30, 100]
    total_experiments = 2000
    total_time_steps = 1000
    total_expected_rewards = []

    for t in temperatures:
        expected_rewards = np.zeros(total_time_steps)
        print(f'Started running experiments for temperature t = {t}')
        for run in range(total_experiments):
            boltz = Boltzmann_exploration(t, total_time_steps)
            boltz()
            expected_rewards = expected_rewards + (boltz.expected_reward - expected_rewards) / (run + 1)
        total_expected_rewards.append(expected_rewards)

    print(f'Finished experiments. Plotting Graphs')
    figure = plt.figure(figsize=(12,8))

    for idx in range(len(temperatures)):
        plt.plot(total_expected_rewards[idx], label=f"t = {temperatures[idx]}")
        plt.legend(loc='lower right')

    plt.xticks([1, 200, 400, 600, 800, 1000])
    plt.xlabel("Time steps")
    plt.ylabel("Expected Reward")
    plt.title(f"Expected Rewards of Boltzmann's action selection after {total_experiments} Experiments for different temperatures values {temperatures}")

    print('Finished plotting')
    plt.show()

    print('Saving Plots')
    plots_dir = os.path.join(os.pardir, 'plots')
    plot_file_name = 'boltzmann_exploration.png'
    if not os.path.isdir(plots_dir):
        os.makedirs(os.path.join(os.pardir, 'plots'))
    figure.savefig(os.path.join(plots_dir, plot_file_name))
    print('Successfully saved Plots')
