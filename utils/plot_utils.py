import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_state_values(q_values, title="Flappy Bird State-Value"):

    def get_Z(x, y):
        if (x,y) in q_values:
            # Find key_value pair with highest Q value in the dictionary
            pi = max(q_values[(x, y)], key=q_values[(x, y)].get)
            V = q_values[(x, y)][pi]
            return V
        else:
            return 0

    def get_figure(ax):
        x_range = np.arange(0, 14)
        y_range = np.arange(-11, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([get_Z(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title(title)
    get_figure(ax)
    plt.show()


def plot_policy(q_values, title="Flappy Bird Policy"):
    """
    Plot the policy for the Flappy Bird environment.

    Args:
        q_values (dict): Dictionary of state-action values.
        title (str): Title of the plot.

    Returns:
        None
    """
    def get_Z(x, y):
        if (x, y) in q_values:
            # Find key_value pair with highest Q value in the dictionary
            pi = max(q_values[(x, y)], key=q_values[(x, y)].get)
            return pi
        else:
            # Return value 2 for unexplored states
            return 2

    def get_figure(ax):
        x_range = np.arange(14, 0, -1)
        y_range = np.arange(-11, 12, 1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x, y) for x in x_range] for y in y_range])
        cmap = colors.ListedColormap(["#BEE9E8", "#62B6CB", "#1B4965"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        surf = ax.imshow(Z, cmap=cmap, norm=norm, extent=[0.5, 13.5, -10.5, 10.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        ax.set_xlabel("x_dist_max-x_dist_min+1")
        ax.set_ylabel("y_dist_max-y_dist_min+1")
        ax.grid(color="w", linestyle="-", linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, cmap=cmap, norm=norm, ticks=[0, 1, 2], cax=cax)
        cbar.ax.set_yticklabels(["0 (Idle)", "1 (Flap)", "2 (Unexplored)"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    get_figure(ax)
    plt.show()

def plot_sum_rewards(all_reward_sums, algorithms=["Q-learning", "Expected SARSA"], title="Sum of rewards during episode"):
    """
    Plot the sum of rewards during an episode for each algorithm.

    Args:
        all_reward_sums (dict): Dictionary of sum of rewards during an episode for each algorithm.

    Returns:
        None
    """
    plt.rcParams["figure.figsize"] = (20,10)

    for algorithm in algorithms:
         # Normalize the sum of rewards
        all_reward_sums[algorithm] = (all_reward_sums[algorithm] - np.min(all_reward_sums[algorithm])) / (np.max(all_reward_sums[algorithm]) - np.min(all_reward_sums[algorithm]))
        plt.plot(all_reward_sums[algorithm], label=algorithm)

    plt.xlabel("Episodes")
    plt.ylabel("Normalized sum of\n rewards\n during\n episode",rotation=0, labelpad=40)
    plt.title(title)
    plt.legend()
    plt.show()