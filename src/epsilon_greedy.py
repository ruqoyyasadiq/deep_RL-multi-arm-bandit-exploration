import os
import numpy as np
import matplotlib.pyplot as plt

class Epsilon_greedy:
    def __init__(self, epsilon, steps, k = 10) -> None:
        self.epsilon = epsilon 
        self.arms = k 
        self.steps = steps
        self.action_space = list(range(k))
        self.r_avg = np.random.normal(loc=1.0, scale=1.0, size=k)

    def run_experiments(self):
        r_actual = []
          
        Q_a = np.zeros(self.arms)
        N_a = np.zeros(self.arms)

        for i in range(self.steps):
            p = np.random.uniform(0, 1, 1)
            if p < self.epsilon:
                action = np.random.choice(self.action_space)
            else:
                action = np.argmax(Q_a)
            
            reward = np.random.normal(self.r_avg[action],1)
            r_actual.append(reward)

            N_a[action] += 1
            Q_a[action] += 1/N_a[action] * (reward - Q_a[action])

        return r_actual

    def __call__(self):
        self.run_experiments()


if __name__ == '__main__':
    arms = 10
    steps = 1000
    runs = 2000
    avg_reward = []
    epsilons = [0, 0.001, 0.01, 0.1, 1]


    for epsilon in epsilons:
        expected_reward = np.zeros(steps)
        for run in range(runs):
            e_greedy = Epsilon_greedy(epsilon, steps, arms)
            expected_reward += e_greedy.run_experiments()

        avg_reward.append(expected_reward/runs)


    n_steps = list(range(steps))
    figure = plt.figure(figsize=(10, 6))
    for i in range(len(epsilons)):
        l = "$\\epsilon$" + "=" + str(epsilons[i])
        plt.plot(n_steps, avg_reward[i], label=l, )

    plt.xticks([1, 200, 400, 600, 800, 1000])
    plt.xlabel("Steps")
    plt.ylabel("Expected reward")
    plt.title(f"Expected Rewards for ɛ-greedy after {runs} experiments for epsilon values {epsilons}")
    plt.legend()
    plt.show()

    # Save Plots
    print('Saving Plots')
    plots_dir = os.path.join(os.pardir, 'plots')
    plot_file_name = 'epsilon_greedy.png'
    if not os.path.isdir(plots_dir):
        os.makedirs(os.path.join(os.pardir, 'plots'))

    print(f'plot_path >>>>> {os.path.join(plots_dir, plot_file_name)}')
    figure.savefig(os.path.join(plots_dir, plot_file_name))
    print('Successfully saved Plots')
