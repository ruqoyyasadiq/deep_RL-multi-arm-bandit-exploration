import os
import numpy as np
import matplotlib.pyplot as plt

class UCB_exploration:
    def __init__(self, c, total_steps, k = 10) -> None:
        self.c = c # set control constant
        self.k = k # number of bandits/actions
        self.Q = np.zeros(k) # value estimates for each action
        self.k_count = np.ones(k) # initialize the count for each action to 1 to prevent zero_divide
        self.t = 1 # time step count
        self.total_steps = total_steps
        self.action_value = np.random.normal(1, 1, k) # sample action value for each action
        self.actual_rewards = np.zeros(total_steps) # this reward is sampled from N(action_value, 1) for each timestep

    def get_action(self) -> int:
        return np.argmax((self.Q + (self.c * np.sqrt(np.log(self.t) / self.k_count)))) # [0, 1, ...9]

    def get_reward_and_estimate_value(self) -> None:
        action = self.get_action()
        reward = np.random.normal(self.action_value[action], 1)
        self.actual_rewards[self.t - 1] = reward

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

    # def __call__(self):
    #     self.run_experiments()


if __name__ == '__main__':
    control = [0, 1, 2, 5]
    total_experiments = 2000
    total_time_steps = 1000
    total_expected_rewards = []

    for c in control:
        expected_rewards = np.zeros(total_time_steps)
        print(f'Started running experiments for ccntrol c = {c}')
        for run in range(total_experiments):
            ucb = UCB_exploration(c, total_time_steps)
            ucb.run_experiments()
            expected_rewards = expected_rewards + (ucb.actual_rewards - expected_rewards) / (run + 1)
        total_expected_rewards.append(expected_rewards)

    print('Finished experiments. Plotting Graphs')
    figure = plt.figure(figsize=(12,8))
    for idx in range(len(control)):
        plt.plot(total_expected_rewards[idx], label=f"c = {control[idx]}")
        plt.legend(loc='lower right')

    plt.xticks([1, 200, 400, 600, 800, 1000])
    plt.xlabel("Time steps")
    plt.ylabel("Expected Reward")
    plt.title(f"Expected Rewards of UCB action selection after {total_experiments} Experiments for different control values {control}")

    print('Finished plotting')
    plt.show()


    # Save Plots
    print('Saving Plots')
    plots_dir = os.path.join(os.pardir, 'plots')
    plot_file_name = 'ucb_exploration.png'
    if not os.path.isdir(plots_dir):
        os.makedirs(os.path.join(os.pardir, 'plots'))
    print(f'plot_path >>>>> {os.path.join(plots_dir, plot_file_name)}')
    figure.savefig(os.path.join(plots_dir, plot_file_name))
    print('Successfully saved Plots')
