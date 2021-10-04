import os
import numpy as np
import matplotlib.pyplot as plt

from ucb_exploration import UCB_exploration
from boltzmann_exploration import Boltzmann_exploration
from epsilon_greedy import Epsilon_greedy
from optimistic_initialization import Optimistic_initialization

def make_comparison():
    total_experiments = 2000
    total_steps = 1000
    best_ucb_control = 2
    best_temperature = 3
    best_epsilon = 0.1
    best_initialization = 10
    # ucb = UCB_exploration(best_ucb_control, total_steps)
    # boltzmann = Boltzmann_exploration(best_temperature, total_steps)
    # eps_greedy = epsilon_greedy_exploration(best_epsilon)
    strategies = {
        'UCB': {
            'name': UCB_exploration,
            'constant': 2
        },
        'Boltzmann': {
            'name': Boltzmann_exploration,
            'constant': 3
        },
        'ε-greedy': {
            'name': Epsilon_greedy,
            'constant': 0.1
        },
        'Optimistic': {
            'name': Optimistic_initialization,
            'constant': 10
        }
        # 'ε-greedy': eps_greedy
        # 'Optimistic Initialization': optimistic,
    }
    total_expected_rewards = []

    for name, strategy in strategies.items():
        expected_rewards = np.zeros(total_steps)
        print(f'Working on {name} exploration')
        if name in ['UCB', 'Boltzmann']:
            for experiment in range(total_experiments):
                expl = strategy['name'](strategy['constant'], total_steps)
                expl.run_experiments()
                expected_rewards = expected_rewards + (expl.actual_rewards - expected_rewards) / (experiment + 1)
            total_expected_rewards.append(expected_rewards)
        else:
            # expected_reward = np.zeros(steps)
            for experiment in range(total_experiments):
                expl = strategy['name'](strategy['constant'], total_steps)
                t = expl.run_experiments()
                expected_rewards = expected_rewards + (t - expected_rewards) / (experiment + 1)
            total_expected_rewards.append(expected_rewards)

    strategy_list = list(strategies.keys())
    
    figure = plt.figure(figsize=(12,8))
    for idx in range(len(total_expected_rewards)):
        if strategy_list[idx] == 'UCB':
            plt.plot(total_expected_rewards[idx], label=f"c = {best_ucb_control}")
        elif strategy_list[idx] == 'Boltzmann':
            plt.plot(total_expected_rewards[idx], label=f"t = {best_temperature}")
        elif strategy_list[idx] == 'ε-greedy':
            plt.plot(total_expected_rewards[idx], label=f"ε = {best_epsilon}")
        elif strategy_list[idx] == 'Optimistic':
            plt.plot(total_expected_rewards[idx], label=f"Q\N{SUBSCRIPT ONE}(a) = {best_initialization}")

    plt.legend(loc='lower right')
    plt.xticks([1, 200, 400, 600, 800, 1000])
    plt.xlabel("Time steps")
    plt.ylabel("Expected Reward")
    plt.title(f"Comparison of Expected Rewards for {total_experiments} experiments for different exploration strategies at c = {best_ucb_control}, t = {best_temperature}, ε = {best_epsilon} and Q\N{SUBSCRIPT ONE}(a) = {best_initialization}")

    plt.show()

    # Save Plots
    print('Saving Plots')
    plots_dir = os.path.join(os.pardir, 'plots')
    plot_file_name = 'comparison_plots.png'
    if not os.path.isdir(plots_dir):
        os.makedirs(os.path.join(os.pardir, 'plots'))
    
    figure.savefig(os.path.join(plots_dir, plot_file_name))
    print('Successfully saved Plots')

if __name__ == '__main__':
    make_comparison()

