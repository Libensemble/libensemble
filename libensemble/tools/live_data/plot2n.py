import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

from libensemble.tools.live_data.live_data import LiveData


def six_hump_camel_func(x1, x2):
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3


class Plot2N(LiveData):
    """
    Plot2N class for generating plots of a 2D domain during a workflow run.

    Parameters
    ----------

    plot_type: str
        Type of plot ("2d" or "3d")

    func: Callable
        Function to plot. Default is six hump camel.

    bounds: tuple
        Bounds for the plot.
    """

    def __init__(self, plot_type="2d", func=six_hump_camel_func, bounds=((-2, 2), (-1, 1.1))):
        """Initialize a new Plot2N class."""
        self.plot_type = plot_type
        self.func = func
        self.grid_x1 = np.linspace(bounds[0][0], bounds[0][1], 50)
        self.grid_x2 = np.linspace(bounds[1][0], bounds[1][1], 50)
        self.last_plot_count = 0
        self.fig = None
        self.ax = None

    def _init_plot(self):
        if self.fig is None:
            plt.ion()  # Enable interactive mode
            self.fig = plt.figure(figsize=(7.3, 5.5))
            if self.plot_type == "3d":
                self.ax = self.fig.add_subplot(111, projection="3d")
            else:
                self.ax = self.fig.add_subplot(111)
                self.first = False
        else:
            self.ax.cla()  # Clear the previous frame
        x1 = self.grid_x1
        x2 = self.grid_x2
        x1, x2 = np.meshgrid(x1, x2)
        f = self.func(x1, x2)

        if self.plot_type == "3d":
            self.ax.plot_surface(x1, x2, f, rstride=1, cstride=1, cmap="winter", edgecolor="none", alpha=0.5, zorder=1)
        else:
            self.ax.contourf(x1, x2, f, cmap="winter")

        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        if self.plot_type == "3d":
            self.ax.set_zlabel("f")

    def _update_plot(self, H):
        self._init_plot()  # need to redo if clearing previous

        self.fig.tight_layout()
        plt.title("Points Selected by APOSMM Local Optimization runs", y=1.0)

        x1 = H["x"][:, 0]
        x2 = H["x"][:, 1]
        x1_locp = H[H["local_pt"]]["x"][:, 0]
        x2_locp = H[H["local_pt"]]["x"][:, 1]
        x1_min = H[H["local_min"]]["x"][:, 0]
        x2_min = H[H["local_min"]]["x"][:, 1]
        num_local_min = len(H[H["local_min"]])

        s = (6, 40, 80)
        c = ("black", "red", "yellow")
        m = ("o", "^", "D")
        lab = ("Point", "Optimization point", "Local minimum")

        common_text_params = {
            "transform": self.ax.transAxes,
            "ha": "left",
            "va": "top",
            "bbox": {"facecolor": "white", "alpha": 0.8},
        }

        text_params = [
            {**common_text_params, "x": 0.05, "y": 0.8, "s": f"Number of local minima found: {num_local_min}"},
            {**common_text_params, "x": 0.6, "y": 0.8, "s": f'Points evaluated: {np.sum(H["sim_ended"])}'},
        ]

        if self.plot_type == "3d":
            f1 = H["f"]
            f2 = H[H["local_pt"]]["f"]
            f3 = H[H["local_min"]]["f"]
            self.ax.scatter3D(x1, x2, f1, s=s[0], color=c[0], marker=m[0], zorder=2, alpha=0.5, label=lab[0])
            self.ax.scatter3D(x1_locp, x2_locp, f2, s=s[1], color=c[1], marker=m[1], zorder=3, label=lab[1])
            self.ax.scatter3D(x1_min, x2_min, f3, s=s[2], color=c[2], marker=m[2], zorder=4, label=lab[2])
            for params in text_params:
                self.ax.text2D(**params)
        else:
            self.ax.scatter(x1, x2, s=s[0], color=c[0], marker=m[0], zorder=2, alpha=0.5, label=lab[0])
            self.ax.scatter(x1_locp, x2_locp, s=s[1], color=c[1], marker=m[1], zorder=3, label=lab[1])
            self.ax.scatter(x1_min, x2_min, s=s[2], color=c[2], marker=m[2], zorder=4, label=lab[2])
            for params in text_params:
                self.ax.text(**params)

        self.ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))
        display(plt.gcf())
        clear_output(wait=True)

    def live_update(self, hist):
        """Function called after every f update in the manager"""
        update_plot_now = False
        if np.count_nonzero(hist.H[self.last_plot_count :]["local_pt"]) > 5:
            update_plot_now = True

        if np.any(hist.H[self.last_plot_count :]["local_min"]):
            update_plot_now = True

        if hist.sim_ended_count >= self.last_plot_count + 50:
            update_plot_now = True

        if update_plot_now:
            self._update_plot(hist.H)
            self.last_plot_count = hist.sim_ended_count

    def finalize(self, hist):
        plt.ioff()  # Disable interactive mode
