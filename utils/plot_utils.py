import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_state_values(V):
    def get_Z(x, y):
        if (x, y) in V:
            return V[x, y]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array(
            [get_Z(x, y, usable_ace) for x, y in zip(np.ravel(X), np.ravel(Y))]
        ).reshape(X.shape)

        surf = ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0
        )
        ax.set_xlabel("Player's Current Sum")
        ax.set_ylabel("Dealer's Showing Card")
        ax.set_zlabel("State Value")
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection="3d")
    ax.set_title("Usable Ace")
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection="3d")
    ax.set_title("No Usable Ace")
    get_figure(False, ax)
    plt.show()


def plot_policy(q_values):
    def get_Z(x, y):
        if q_values[(x, y)]:
            # Find key_value pair with highest Q value in the dictionary
            pi = max(q_values[(x, y)], key=q_values[(x, y)].get)
            return pi
        else:
            return 2

    def get_figure(ax):
        x_range = np.arange(0, 14, -1)
        y_range = np.arange(-11, 11, 1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x, y) for x in x_range] for y in y_range])
        surf = ax.imshow(
            Z,
            cmap=plt.get_cmap("Pastel2", 2),
            vmin=0,
            vmax=2,
            extent=[0.5, 13.5, -10.5, 10.5],
        )
        plt.xticks(x_range)
        plt.yticks(y_range)
        plt.gca().invert_yaxis()
        ax.set_xlabel("x_dist_max-x_dist_min+1")
        ax.set_ylabel("y_dist_max-y_dist_min+1")
        ax.grid(color="w", linestyle="-", linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(["0 (Idle)", "1 (Flap)"])

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title("Flappy Bird Policy")
    get_figure(ax)
    plt.show()
