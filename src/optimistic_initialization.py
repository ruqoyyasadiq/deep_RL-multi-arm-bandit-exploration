import os
import numpy as np
import matplotlib.pyplot as plt

class Optimistic_initialization:
    def __init__(self, opt_init, steps, k = 10) -> None:
        self.opt_init = opt_init 
        self.arms = k 
        self.steps = steps
        self.action_space = list(range(k))
        self.r_avg = np.random.normal(loc=1.0, scale=1.0, size=k)

    def run_experiments(self): 
        r_actual = []
            
        Q_a = np.ones(self.arms) * self.opt_init
        N_a = np.zeros(self.arms)

        for i in range(self.steps):
            action = np.argmax(Q_a)
                
            reward = np.random.normal(self.r_avg[action],1)
            r_actual.append(reward)

            N_a[action] += 1
            Q_a[action] += 1/N_a[action] * (reward - Q_a[action])

        return r_actual

if __name__ == '__main__':
    arms = 10
    steps = 1000
    runs = 2000
    avg_reward = []
    optimistic_init = [0, 1, 2, 5, 10]

    for init in optimistic_init:
        expected_reward = np.zeros(steps)
        for run in range(runs):
            o_init = Optimistic_initialization(init, steps, arms)
            expected_reward += o_init.run_experiments()

        avg_reward.append(expected_reward/runs)


    n_steps = list(range(steps))
    figure = plt.figure(figsize=(10, 6))
    for i in range(len(optimistic_init)):
        l = 'Q\N{SUBSCRIPT ONE}(a)'+"="+str(optimistic_init[i])
        plt.plot(n_steps, avg_reward[i], label=l, )

    plt.xlabel("Steps")
    plt.ylabel("Expected reward")
    plt.title(f"Expected Rewards for optimistic initialization after {runs} experiments for init values {optimistic_init}")
    plt.legend()
    plt.show()


    # Save Plots
    print('Saving Plots')
    plots_dir = os.path.join(os.pardir, 'plots')
    plot_file_name = 'optimistic_initialization.png'
    if not os.path.isdir(plots_dir):
        os.makedirs(os.path.join(os.pardir, 'plots'))
    figure.savefig(os.path.join(plots_dir, plot_file_name))
    print('Successfully saved Plots')