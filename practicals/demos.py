import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from ipywidgets import interactive, FloatSlider, FloatLogSlider, Checkbox
from dataclasses import dataclass, field
from typing import Tuple, Callable
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.datasets import make_circles

@dataclass
class LinearRegression1D:
    seed: int = 42
    x_range: Tuple[float, float] = (-5, 5)
    num_points: int = 100
    theta_0: float = 1
    theta_1: float = 1.5
    noise: float = 2
    scatter_color: str = 'red'
    line_color: str = 'blue'
    error_color: str = 'green'
    theta_range: Tuple[float, float] = (-10, 10)
    step: float = 0.1
    plot_margin: float = 1
    loss_resolution: int = 50
    
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    t0_mesh: np.ndarray = field(init=False)
    t1_mesh: np.ndarray = field(init=False)
    mse_vals: np.ndarray = field(init=False)

    def __post_init__(self):
        np.random.seed(self.seed)
        self.x, self.y = self._generate_data()
        self._prepare_loss_meshgrid()

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(*self.x_range, self.num_points)
        y = self.theta_0 + self.theta_1 * x + self.noise * np.random.randn(*x.shape)
        return x, y

    def _prepare_loss_meshgrid(self):
        theta_0_vals = np.linspace(*self.theta_range, self.loss_resolution)
        theta_1_vals = np.linspace(*self.theta_range, self.loss_resolution)
        self.t0_mesh, self.t1_mesh = np.meshgrid(theta_0_vals, theta_1_vals)

        y_pred_matrix = self.t0_mesh[..., np.newaxis] + self.t1_mesh[..., np.newaxis] * self.x
        self.mse_vals = np.mean((self.y - y_pred_matrix) ** 2, axis=2)

    def _plot_regression(self, ax: plt.Axes, theta_0: float, theta_1: float):
        y_pred = theta_0 + theta_1 * self.x 
        mse = np.mean((self.y - y_pred) ** 2)
        
        ax.scatter(self.x, self.y, color=self.scatter_color, label='Data points')
        ax.plot(self.x, y_pred, color=self.line_color, label=f'Regression line, MSE: {mse:.2f}')
        ax.vlines(self.x, ymin=self.y, ymax=y_pred, color=self.error_color, linestyle='dashed')
        
        ax.set_xlim(self.x_range[0] - self.plot_margin, self.x_range[1] + self.plot_margin)
        ax.set_ylim(self.y.min() - self.plot_margin, self.y.max() + self.plot_margin)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Linear regression demo')
        ax.legend()

    def _plot_loss_function(self, ax: plt.Axes, theta_0: float, theta_1: float):
        contour = ax.contour(self.t0_mesh, self.t1_mesh, self.mse_vals, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='MSE')
        ax.scatter([theta_0], [theta_1], color='red', marker='x', s=100, label='Current parameters')
        
        ax.set_xlabel('$\\theta_0$')
        ax.set_ylabel('$\\theta_1$')
        ax.set_title('Loss function (MSE)')
        ax.legend()

    def plot(self, theta_0: float, theta_1: float):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        self._plot_regression(ax1, theta_0, theta_1)
        self._plot_loss_function(ax2, theta_0, theta_1)
        plt.show()

    def run(self):
        theta_0_slider = FloatSlider(
            value=self.theta_0, min=self.theta_range[0], max=self.theta_range[1], step=self.step, 
            description='$\\theta_0$ (bias):'
        )
        theta_1_slider = FloatSlider(
            value=self.theta_1, min=self.theta_range[0], max=self.theta_range[1], step=self.step, 
            description='$\\theta_1$ (weight):'
        )
        interactive_plot = interactive(self.plot, theta_0=theta_0_slider, theta_1=theta_1_slider)
        output = interactive_plot.children[-1]
        output.layout.height = '400px'
        return interactive_plot


@dataclass
class LinearRegression2D:
    seed: int = 42
    x_range: Tuple[float, float] = (-5, 5)
    grid_points: int = 5
    theta_0: float = 1
    theta_1: float = 1.5
    theta_2: float = 2
    noise: float = 2
    scatter_color: str = 'red'
    surface_color: str = 'blue'
    error_color: str = 'green'
    theta_range: Tuple[float, float] = (-10, 10)
    step: float = 0.1
    plot_margin: float = 1
    
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self):
        np.random.seed(self.seed)
        self.x, self.y = self._generate_data()

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x1, x2 = np.meshgrid(np.linspace(*self.x_range, self.grid_points), np.linspace(*self.x_range, self.grid_points))
        x = np.vstack((x1.flatten(), x2.flatten())).T
        y = self.theta_0 + self.theta_1 * x[:, 0] + self.theta_2 * x[:, 1] + self.noise * np.random.randn(x.shape[0])
        return x, y

    def _plot_regression(self, ax: plt.Axes, theta_0: float, theta_1: float, theta_2: float):
        y_pred = theta_0 + theta_1 * self.x[:, 0] + theta_2 * self.x[:, 1]
        mse = np.mean((self.y - y_pred) ** 2)

        x1_plane, x2_plane = np.meshgrid(np.linspace(*self.x_range, 10), np.linspace(*self.x_range, 10))
        y_plane = theta_1 * x1_plane + theta_2 * x2_plane + theta_0
        
        ax.scatter(self.x[:, 0], self.x[:, 1], self.y, color=self.scatter_color, label='Data points')
        ax.plot_surface(x1_plane, x2_plane, y_plane, color=self.surface_color, alpha=0.5, label=f'Regression plane, MSE: {mse:.2f}')
        for i in range(len(self.x)):
            ax.plot(
                [self.x[i, 0], self.x[i, 0]], [self.x[i, 1], self.x[i, 1]], [self.y[i], y_pred[i]], 
                color=self.error_color, linestyle='dashed'
            )

        ax.set_xlim(self.x_range[0] - self.plot_margin, self.x_range[1] + self.plot_margin)
        ax.set_ylim(self.x_range[0] - self.plot_margin, self.x_range[1] + self.plot_margin)
        ax.set_zlim(self.y.min() - self.plot_margin, self.y.max() + self.plot_margin)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$y$')
        ax.set_title('Linear regression demo')
        #ax.legend()#this line of code causes the code to crash due to the fact that it treats the 3d plot as a 2d plot. 
        # use the same colors you used to draw
        plane_color = "royalblue"   # color used in plot_surface(..., color=plane_color, alpha=0.5)
        point_color = "crimson"     # color used in ax.scatter(...)

        # proxy artists for legend (no drawing, just legend handles)
        legend_handles = [
            Patch(facecolor=plane_color, edgecolor=plane_color, alpha=0.5, label=f"Regression plane, MSE: {mse:.2f}"),
            Line2D([0], [0], marker='o', linestyle='None',
                markerfacecolor=point_color, markeredgecolor=point_color, markersize=8,
                label="Data points"),

        ]
        ax.legend(handles=legend_handles, loc="upper left", frameon=True)



    def plot(self, theta_0: float, theta_1: float, theta_2: float):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        self._plot_regression(ax, theta_0, theta_1, theta_2)
        plt.show()

    def run(self):
        theta_0_slider = FloatSlider(
            value=self.theta_0, min=self.theta_range[0], max=self.theta_range[1], step=self.step, 
            description='$\\theta_0$ (bias):'
        )
        theta_1_slider = FloatSlider(
            value=self.theta_1, min=self.theta_range[0], max=self.theta_range[1], step=self.step, 
            description='$\\theta_1$ (weight for $x_1$):'
        )
        theta_2_slider = FloatSlider(
            value=self.theta_2, min=self.theta_range[0], max=self.theta_range[1], step=self.step, 
            description='$\\theta_2$ (weight for $x_2$):'
        )
        interactive_plot = interactive(self.plot, theta_0=theta_0_slider, theta_1=theta_1_slider, theta_2=theta_2_slider)
        output = interactive_plot.children[-1]
        output.layout.height = '550px'
        return interactive_plot


@dataclass
class LogisticRegression1D:
    seed: int = 42
    x_range: Tuple[float, float] = (-5, 5)
    num_points: int = 100
    theta_0: float = 1
    theta_1: float = 1.5
    scatter_color: str = 'red'
    line_color: str = 'blue'
    error_color: str = 'green'
    decision_boundary_color: str = 'orange'
    theta_range: Tuple[float, float] = (-10, 10)
    step: float = 0.1
    plot_margin: float = 1
    loss_resolution: int = 50

    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    t0_mesh: np.ndarray = field(init=False)
    t1_mesh: np.ndarray = field(init=False)
    nll_vals: np.ndarray = field(init=False)

    def __post_init__(self):
        np.random.seed(self.seed)
        self.x, self.y = self._generate_data()
        self._prepare_loss_meshgrid()

    @staticmethod
    def _logistic_1d(x: np.ndarray, theta_0: float, theta_1: float) -> np.ndarray:
        return 1 / (1 + np.exp(-(theta_1 * x + theta_0)))

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(*self.x_range, self.num_points)
        prob = self._logistic_1d(x, self.theta_0, self.theta_1)
        y = np.random.binomial(1, prob)
        return x, y

    def _prepare_loss_meshgrid(self):
        theta_0_vals = np.linspace(*self.theta_range, self.loss_resolution)
        theta_1_vals = np.linspace(*self.theta_range, self.loss_resolution)
        self.t0_mesh, self.t1_mesh = np.meshgrid(theta_0_vals, theta_1_vals)
        
        y_pred_matrix = self._logistic_1d(self.x[:, np.newaxis, np.newaxis], self.t0_mesh, self.t1_mesh)        
        y_pred_matrix_clipped = np.clip(y_pred_matrix, 1e-15, 1 - 1e-15) 
        self.nll_vals = -np.mean(
            self.y[:, np.newaxis, np.newaxis] * np.log(y_pred_matrix_clipped) +
            (1 - self.y[:, np.newaxis, np.newaxis]) * np.log(1 - y_pred_matrix_clipped), axis=0
        )

    def _plot_regression(self, ax: plt.Axes, theta_0: float, theta_1: float):
        y_pred_prob = self._logistic_1d(self.x, theta_0, theta_1)        
        y_pred_prob_clipped = np.clip(y_pred_prob, 1e-15, 1 - 1e-15)
        nll = -np.mean(self.y * np.log(y_pred_prob_clipped) + (1 - self.y) * np.log(1 - y_pred_prob_clipped))

        decision_boundary = -theta_0 / (theta_1 + 1e-5)
        
        ax.scatter(self.x, self.y, color=self.scatter_color, label='Data points')
        ax.plot(self.x, y_pred_prob, color=self.line_color, label=f'Logistic regression curve (NLL = {nll:.2f})')
        ax.vlines(self.x, ymin=self.y, ymax=y_pred_prob, color=self.error_color, linestyle='dashed', label='Errors')
        ax.axvline(x=decision_boundary, color=self.decision_boundary_color, linestyle='--', label='Decision boundary')
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('Probability')        
        ax.set_title('Logistic regression demo')
        ax.legend()

    def _plot_loss_function(self, ax: plt.Axes, theta_0: float, theta_1: float):
        contour = ax.contour(self.t0_mesh, self.t1_mesh, self.nll_vals, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='NLL')
        ax.scatter([theta_0], [theta_1], color='red', marker='x', s=100, label='Current parameters')
        
        ax.set_xlabel('$\\theta_0$')
        ax.set_ylabel('$\\theta_1$')
        ax.set_title('Loss function (NLL)')
        ax.legend()

    def plot(self, theta_0: float, theta_1: float):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        self._plot_regression(ax1, theta_0, theta_1)
        self._plot_loss_function(ax2, theta_0, theta_1)
        plt.show()

    def run(self):
        theta_0_slider = FloatSlider(
            value=self.theta_0, min=self.theta_range[0], max=self.theta_range[1], step=self.step, 
            description='$\\theta_0$ (bias):'
        )
        theta_1_slider = FloatSlider(
            value=self.theta_1, min=self.theta_range[0], max=self.theta_range[1], step=self.step, 
            description='$\\theta_1$ (weight):'
        )
        interactive_plot = interactive(self.plot, theta_0=theta_0_slider, theta_1=theta_1_slider)
        output = interactive_plot.children[-1]
        output.layout.height = '400px'
        return interactive_plot


@dataclass
class LogisticRegression2D:
    seed: int = 42
    x_range: Tuple[float, float] = (-5, 5)
    grid_points: int = 5
    noise_intensity: float = 0.5
    theta_0: float = 1
    theta_1: float = 1.5
    theta_2: float = 2
    scatter_color: str = 'red'
    surface_color: str = 'blue'
    error_color: str = 'green'
    decision_boundary_color: str = 'orange'
    cmap_colors: Tuple[str, str] = ('pink', 'darkred')
    plot_margin_percent: float = 0.1 

    x1: np.ndarray = field(init=False)
    x2: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    x1_range: Tuple[float, float] = field(init=False)
    x2_range: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        np.random.seed(self.seed)
        self.x1, self.x2, self.y = self._generate_data()
        self._calculate_ranges()

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x1, x2 = np.meshgrid(np.linspace(*self.x_range, self.grid_points), np.linspace(*self.x_range, self.grid_points))
        x1 = x1.flatten() + np.random.normal(0, self.noise_intensity, x1.size)
        x2 = x2.flatten() + np.random.normal(0, self.noise_intensity, x2.size)
        prob = self._logistic_2d(x1, x2, self.theta_0, self.theta_1, self.theta_2)
        y = np.random.binomial(1, prob)
        return x1, x2, y

    def _calculate_ranges(self):
        x1_min, x1_max = self.x1.min(), self.x1.max()
        x2_min, x2_max = self.x2.min(), self.x2.max()

        x1_padding = (x1_max - x1_min) * self.plot_margin_percent
        x2_padding = (x2_max - x2_min) * self.plot_margin_percent

        self.x1_range = (x1_min - x1_padding, x1_max + x1_padding)
        self.x2_range = (x2_min - x2_padding, x2_max + x2_padding)

    @staticmethod
    def _logistic_2d(x1: np.ndarray, x2: np.ndarray, theta_0: float, theta_1: float, theta_2: float) -> np.ndarray:
        z = theta_0 + theta_1 * x1 + theta_2 * x2
        return 1 / (1 + np.exp(-z))

    def _plot_regression(self, ax: plt.Axes, theta_0: float, theta_1: float, theta_2: float, show_errors: bool, show_legend: bool):
        y_pred_prob = self._logistic_2d(self.x1, self.x2, theta_0, theta_1, theta_2)
        y_pred_prob_clipped = np.clip(y_pred_prob, 1e-15, 1 - 1e-15)
        nll = -np.mean(self.y * np.log(y_pred_prob_clipped) + (1 - self.y) * np.log(1 - y_pred_prob_clipped))

        x1_range = np.linspace(*self.x1_range, 100)
        x2_range = np.linspace(*self.x2_range, 100)
        x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
        y_pred_mesh = self._logistic_2d(x1_mesh, x2_mesh, theta_0, theta_1, theta_2)

        ax.scatter(self.x1, self.x2, self.y, color=self.scatter_color)
        ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, color=self.surface_color, alpha=0.3)
        ax.contour(x1_mesh, x2_mesh, y_pred_mesh, levels=[0.5], colors=self.decision_boundary_color, linestyles='--')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Probability')
        ax.set_title(f'Logistic regression demo (NLL = {nll:.2f})')

        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.scatter_color, markersize=10, label='Data points'),
            Patch(color=self.surface_color, alpha=0.3, label='Regression surface'),
            Line2D([0], [0], linestyle='--', color=self.decision_boundary_color, label='Decision boundary')
        ]

        if show_errors:
            for i in range(len(self.x1)):
                ax.plot(
                    [self.x1[i], self.x1[i]],
                    [self.x2[i], self.x2[i]],
                    [self.y[i], y_pred_prob[i]],
                    color=self.error_color,
                    linestyle='dashed'
                )
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color=self.error_color,
                    linestyle='dashed',
                    label=f'Errors (NLL = {nll:.2f})'
                )
            )

        if show_legend:
            ax.legend(handles=legend_handles)

    def _plot_2d(self, ax: plt.Axes, theta_0: float, theta_1: float, theta_2: float):
        x1_range = np.linspace(*self.x1_range, 100)
        x2_range = np.linspace(*self.x2_range, 100)
        x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
        y_pred_mesh = self._logistic_2d(x1_mesh, x2_mesh, theta_0, theta_1, theta_2)
        
        custom_cmap = ListedColormap(list(self.cmap_colors))
        scatter = ax.scatter(self.x1, self.x2, c=self.y, cmap=custom_cmap)
        ax.contour(x1_mesh, x2_mesh, y_pred_mesh, levels=[0.5], colors=self.decision_boundary_color, linestyles='--')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Decision boundary in 2D')

        ax.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.cmap_colors[0], markersize=10, label='Class 0'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.cmap_colors[1], markersize=10, label='Class 1'),
            Line2D([0], [0], linestyle='--', color=self.decision_boundary_color, label='Decision boundary')
        ])

    def plot(self, theta_0: float, theta_1: float, theta_2: float, show_errors: bool = True, show_legend: bool = True, show_2d: bool = True):
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_regression(ax1, theta_0, theta_1, theta_2, show_errors, show_legend)
        if show_2d:
            ax2 = fig.add_subplot(122)
            self._plot_2d(ax2, theta_0, theta_1, theta_2)
        plt.show()

    def run(self):
        theta_0_slider = FloatSlider(value=self.theta_0, min=-10.0, max=10.0, step=0.1, description='$\\theta_0$ (bias):')
        theta_1_slider = FloatSlider(value=self.theta_1, min=-10.0, max=10.0, step=0.1, description='$\\theta_1$ (weight for x1):')
        theta_2_slider = FloatSlider(value=self.theta_2, min=-10.0, max=10.0, step=0.1, description='$\\theta_2$ (weight for x2):')
        show_errors = Checkbox(value=True, description='Show errors')
        show_legend = Checkbox(value=True, description='Show legend')
        show_2d = Checkbox(value=True, description='Show 2D')

        interactive_plot = interactive(
            self.plot,
            theta_0=theta_0_slider,
            theta_1=theta_1_slider,
            theta_2=theta_2_slider,
            show_errors=show_errors,
            show_legend=show_legend,
            show_2d=show_2d
        )
        output = interactive_plot.children[-1]
        output.layout.height = '500px'
        return interactive_plot


@dataclass
class RegularizedRegression:
    penalty_type: str = 'ridge'
    seed: int = 42
    alpha: float = 1.0
    theta_range: Tuple[float, float] = (-10, 10)
    grid_size: int = 99
    contour_mse_color: str = 'viridis'
    contour_penalty_color: str = 'plasma'
    mse_cutoff: float = 10
    penalty_cutoff: float = 10
    num_points: int = 100
    x_range: Tuple[float, float] = (0, 2)
    theta_0: float = 5
    theta_1: float = 2.5
    noise: float = 1.0

    X: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    t0_mesh: np.ndarray = field(init=False)
    t1_mesh: np.ndarray = field(init=False)
    mse_vals: np.ndarray = field(init=False)
    penalty_vals: np.ndarray = field(init=False)
    total_loss_vals: np.ndarray = field(init=False)

    def __post_init__(self):
        np.random.seed(self.seed)
        self.X, self.y = self._generate_data()
        self._prepare_meshgrid()
        self._calculate_static_loss_values()

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X = (self.x_range[1] - self.x_range[0]) * np.random.rand(self.num_points, 2) + self.x_range[0]
        y = self.theta_0 * X[:, 0] + self.theta_1 * X[:, 1] + self.noise * np.random.randn(self.num_points)
        return X, y

    def _prepare_meshgrid(self):
        theta_0_vals = np.linspace(*self.theta_range, self.grid_size)
        theta_1_vals = np.linspace(*self.theta_range, self.grid_size)
        self.t0_mesh, self.t1_mesh = np.meshgrid(theta_0_vals, theta_1_vals)

    def _calculate_static_loss_values(self):
        theta_0_flat = self.t0_mesh.ravel()
        theta_1_flat = self.t1_mesh.ravel()

        predictions = theta_0_flat[:, np.newaxis] * self.X[:, 0] + theta_1_flat[:, np.newaxis] * self.X[:, 1]
        errors = predictions - self.y
        mse_flat = np.mean(errors**2, axis=1)
        self.mse_vals = mse_flat.reshape(self.t0_mesh.shape)

        if self.penalty_type == 'ridge':
            penalty_flat = theta_0_flat**2 + theta_1_flat**2
        elif self.penalty_type == 'lasso':
            penalty_flat = np.abs(theta_0_flat) + np.abs(theta_1_flat)

        self.penalty_vals = penalty_flat.reshape(self.t0_mesh.shape)

    def _calculate_total_loss_values(self, alpha: float):
        self.total_loss_vals = self.mse_vals + alpha * self.penalty_vals

    def _plot_contours(self, ax: plt.Axes, alpha: float):
        self._calculate_total_loss_values(alpha)

        contour_mse = ax.contour(
            self.t0_mesh, self.t1_mesh, np.clip(self.mse_vals, None, self.mse_cutoff), 
            levels=10, cmap=self.contour_mse_color, linestyles='dotted'
        )
        plt.colorbar(contour_mse, ax=ax, label='MSE Loss')

        contour_penalty = ax.contour(
            self.t0_mesh, self.t1_mesh, np.clip(self.penalty_vals, None, self.penalty_cutoff), 
            levels=10, cmap=self.contour_penalty_color, linestyles='dashed'
        )
        plt.colorbar(contour_penalty, ax=ax, label=f'{self.penalty_type.capitalize()} Penalty')

        min_mse_index = np.unravel_index(np.argmin(self.mse_vals, axis=None), self.mse_vals.shape)
        min_penalty_index = np.unravel_index(np.argmin(self.penalty_vals, axis=None), self.penalty_vals.shape)
        min_total_loss_index = np.unravel_index(np.argmin(self.total_loss_vals, axis=None), self.total_loss_vals.shape)

        ax.plot(
            self.t0_mesh[min_mse_index], self.t1_mesh[min_mse_index], 
            'go', markersize=10, alpha=0.5, label='Minimum of MSE Loss'
        )
        ax.plot(
            self.t0_mesh[min_penalty_index], self.t1_mesh[min_penalty_index], 
            'bo', markersize=10, alpha=0.5, label=f'Minimum of {self.penalty_type.capitalize()} Penalty'
        )
        ax.plot(
            self.t0_mesh[min_total_loss_index], self.t1_mesh[min_total_loss_index], 
            'ro', markersize=10, alpha=0.5, label='Minimum of Combined Loss'
        )

        ax.set_title(f'Isolines of MSE Loss and {self.penalty_type.capitalize()} Penalty with Cutoff')
        ax.set_xlabel('$\\theta_0$')
        ax.set_ylabel('$\\theta_1$')
        ax.legend(loc='lower left')
        ax.grid(True)

    def plot(self, alpha: float):
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_contours(ax, alpha)
        plt.show()

    def run(self):
        alpha_slider = FloatLogSlider(
            value=self.alpha, base=10, min=-3, max=3, step=0.1, 
            description=r'$\alpha$'
        )
        interactive_plot = interactive(self.plot, alpha=alpha_slider)
        output = interactive_plot.children[-1]
        output.layout.height = '600px'
        return interactive_plot


@dataclass
class ANDXORToyDatasets:
    n_samples: int = 100
    noise: float = 0.1
    scatter_color_0: str = 'red'
    scatter_color_1: str = 'blue'
    criterion: Callable = nn.BCELoss()
    learning_rate: float = 0.1
    epochs: int = 10000
    
    X_xor: np.ndarray = field(init=False)
    y_xor: np.ndarray = field(init=False)
    X_and: np.ndarray = field(init=False)
    y_and: np.ndarray = field(init=False)

    def __post_init__(self):
        self.X_xor, self.y_xor = self._generate_data('xor')
        self.X_and, self.y_and = self._generate_data('and')

    def _generate_data(self, operation: str) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        if operation == 'xor':
            y = np.array([0, 1, 1, 0])
        elif operation == 'and':
            y = np.array([0, 0, 0, 1])
        else:
            raise ValueError("Unknown operation: choose 'xor' or 'and'")

        np.random.seed(42)
        X_expanded = np.vstack([x + self.noise * np.random.randn(self.n_samples, 2) for x in X]) - 0.5
        y_expanded = np.hstack([np.full(self.n_samples, label) for label in y])

        return X_expanded, y_expanded

    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].scatter(self.X_and[self.y_and == 0][:, 0], self.X_and[self.y_and == 0][:, 1], color=self.scatter_color_0, label='Class 0')
        axs[0].scatter(self.X_and[self.y_and == 1][:, 0], self.X_and[self.y_and == 1][:, 1], color=self.scatter_color_1, label='Class 1')
        axs[0].set_title('AND Dataset')
        axs[0].legend()

        axs[1].scatter(self.X_xor[self.y_xor == 0][:, 0], self.X_xor[self.y_xor == 0][:, 1], color=self.scatter_color_0, label='Class 0')
        axs[1].scatter(self.X_xor[self.y_xor == 1][:, 0], self.X_xor[self.y_xor == 1][:, 1], color=self.scatter_color_1, label='Class 1')
        axs[1].set_title('XOR Dataset')
        axs[1].legend()

        axs[0].set_xlabel('Feature 1')
        axs[0].set_ylabel('Feature 2')
        axs[1].set_xlabel('Feature 1')
        axs[1].set_ylabel('Feature 2')
        
        plt.show()

    def _train_model(self, X: np.ndarray, y: np.ndarray, model: nn.Module, epochs: int):
        
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        inputs = torch.tensor(X, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        for epoch in range(epochs):            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        return model

    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, model: Callable, ax, title: str):
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 1000), np.linspace(x2_min, x2_max, 1000))
        inputs = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        Z = model(inputs).detach().numpy().reshape(xx.shape)

        custom_cmap = ListedColormap([self.scatter_color_0, self.scatter_color_1])
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=custom_cmap)
        ax.contour(xx, yy, Z, levels=[0.5], colors='orange', linestyles="--", linewidths=2)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=custom_cmap, edgecolor='k', label='Data')

        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])

    def _initialize_model(self, model: nn.Module):
        new_model = copy.deepcopy(model)
        for layer in new_model.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        return new_model

    def run(self, model: nn.Module):                
        model_and = self._initialize_model(model)
        model_and = self._train_model(self.X_and, self.y_and, model_and, self.epochs)

        model_xor = self._initialize_model(model)        
        model_xor = self._train_model(self.X_xor, self.y_xor, model_xor, self.epochs)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        self.plot_decision_boundary(self.X_and, self.y_and, model_and, axs[0], model_and.__class__.__name__ + ' on AND Problem')
        self.plot_decision_boundary(self.X_xor, self.y_xor, model_xor, axs[1], model_xor.__class__.__name__ + ' on XOR Problem')
        plt.show()

        return model_and, model_xor
               

@dataclass
class ToyDataset:
    dataset_type: str
    n_samples: int = 100
    noise: float = 0.1
    scatter_color_0: str = 'red'
    scatter_color_1: str = 'blue'
    criterion: Callable = nn.BCELoss()
    learning_rate: float = 0.1
    epochs: int = 10000
    
    X: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self):
        self.X, self.y = self._generate_data(self.dataset_type)

    def _generate_data(self, operation: str) -> Tuple[np.ndarray, np.ndarray]:
        if operation == 'circles':
            X, y = make_circles(n_samples=self.n_samples, noise=self.noise, factor=0.5, random_state=42)
        elif operation == 'spirals':
            X, y = self._generate_spirals(self.n_samples, self.noise)
        else:
            raise ValueError("Unknown operation: choose 'xor', 'and', 'circles', or 'spirals'")

        if operation in ['xor', 'and']:
            np.random.seed(42)
            X_expanded = np.vstack([x + self.noise * np.random.randn(self.n_samples, 2) for x in X]) - 0.5
            y_expanded = np.hstack([np.full(self.n_samples, label) for label in y])
        else:
            X_expanded, y_expanded = X, y

        return X_expanded, y_expanded

    def _generate_spirals(self, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)
        n = np.sqrt(np.random.rand(n_samples//2)) * 720 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_samples//2) * noise
        d1y = np.sin(n) * n + np.random.rand(n_samples//2) * noise
        X1 = np.vstack([d1x, d1y]).T
        
        d2x = np.cos(n) * n + np.random.rand(n_samples//2) * noise
        d2y = -np.sin(n) * n + np.random.rand(n_samples//2) * noise
        X2 = np.vstack([d2x, d2y]).T
        
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        return X, y

    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 5))
        
        ax.scatter(self.X[self.y == 0][:, 0], self.X[self.y == 0][:, 1], color=self.scatter_color_0, label='Class 0')
        ax.scatter(self.X[self.y == 1][:, 0], self.X[self.y == 1][:, 1], color=self.scatter_color_1, label='Class 1')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'{self.dataset_type.upper()} Dataset')

        ax.legend()
        plt.show()

    def _train_model(self, X: np.ndarray, y: np.ndarray, model: nn.Module, epochs: int):
        
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        inputs = torch.tensor(X, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        for epoch in range(epochs):            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        return model

    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, model: Callable, ax, title: str):
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 1000), np.linspace(x2_min, x2_max, 1000))
        inputs = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        Z = model(inputs).detach().numpy().reshape(xx.shape)

        custom_cmap = ListedColormap([self.scatter_color_0, self.scatter_color_1])
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=custom_cmap)
        ax.contour(xx, yy, Z, levels=[0.5], colors='orange', linestyles="--", linewidths=2)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=custom_cmap, edgecolor='k', label='Data')

        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])

    def _initialize_model(self, model: nn.Module):
        new_model = copy.deepcopy(model)
        for layer in new_model.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        return new_model

    def run(self, model: nn.Module):                
        model_initialized = self._initialize_model(model)
        nn_model = self._train_model(self.X, self.y, model_initialized, self.epochs)

        fig, ax = plt.subplots(figsize=(6, 5))
        self.plot_decision_boundary(self.X, self.y, nn_model, ax, model_initialized.__class__.__name__ + f' on {self.dataset_type.upper()} Problem')
        plt.show()

