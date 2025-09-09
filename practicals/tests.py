import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import mean_squared_error, log_loss
from matplotlib.colors import ListedColormap 
import torch

def create_mesh_grid(X, padding=1, resolution=1000):
    x1_min, x1_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    x2_min, x2_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    x1_mesh, x2_mesh = np.meshgrid(
        np.linspace(x1_min, x1_max, resolution), 
        np.linspace(x2_min, x2_max, resolution)
    )
    x_mesh = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
    return x1_mesh, x2_mesh, x_mesh


def generate_regression_data(n_samples=100, n_features=1, noise=10, random_state=42):
    return make_regression(
        n_samples=n_samples, 
        n_features=n_features, 
        noise=noise, 
        random_state=random_state
    )


def generate_classification_data(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_classes=2, 
    n_clusters_per_class=2, random_state=42
):
    return make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative, 
        n_redundant=n_redundant, 
        n_classes=n_classes, 
        n_clusters_per_class=n_clusters_per_class, 
        random_state=random_state
    )


def plot_regression(X, y, predictions, labels, scatter_color='red', colors=['green', 'blue', 'red'], line_widths=[8, 4, 2], alphas=[0.4, 0.5, 0.6]):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color=scatter_color, label='Data')
    for y_pred, color, label, lw, alpha in zip(predictions, colors, labels, line_widths, alphas):
        plt.plot(X, y_pred, label=label, color=color, linewidth=lw, alpha=alpha)
    plt.xlabel('Feature x')
    plt.ylabel('Target y')
    plt.title('Comparison of linear regression implementations')
    plt.legend()
    plt.show()


def linear_regression(
    CustomLinearModel, ReferenceLinearModel, alpha=None, epochs=1000, learning_rate=0.01,
    scatter_color='red', colors=['green', 'blue', 'red'], line_widths=[8, 4, 2], alphas=[0.4, 0.5, 0.6]
):
    X, y = generate_regression_data()
    
    if alpha is not None:
        model_normal = CustomLinearModel(solver='normal', alpha=alpha)
        model_gd = CustomLinearModel(
            solver='gd', alpha=alpha, epochs=epochs, learning_rate=learning_rate
        )
        reference_model = ReferenceLinearModel(alpha=alpha)
    else:
        model_normal = CustomLinearModel(solver='normal')
        model_gd = CustomLinearModel(
            solver='gd', epochs=epochs, learning_rate=learning_rate
        )
        reference_model = ReferenceLinearModel()
        
    models = [model_normal, model_gd, reference_model]
    
    for model in models:
        model.fit(X, y)
        
    predictions = [model.predict(X) for model in models]
    mses = [mean_squared_error(y, y_pred) for y_pred in predictions]
    labels = [f'Reference model, MSE: {mses[2]:.2f}',
              f'Least squares, MSE: {mses[0]:.2f}',
              f'GD, MSE: {mses[1]:.2f}'
             ]
    
    plot_regression(X, y, predictions, labels)


def logistic_regression(
    CustomLogisticModel, ReferenceLogisticModel, alpha=None, epochs=1000, learning_rate=0.01,
    cmap_colors=('pink', 'darkred'), ref_color='green', custom_color='orange'
):
    X, y = generate_classification_data()
    
    if alpha is not None:
        model_custom = CustomLogisticModel(
            epochs=epochs, learning_rate=learning_rate, alpha=alpha
        )
        reference_model = ReferenceLogisticModel(alpha)
    else:
        model_custom = CustomLogisticModel(epochs=epochs, learning_rate=learning_rate)
        reference_model = ReferenceLogisticModel()
        
    models = [model_custom, reference_model]
    for model in models:
        model.fit(X, y)

    y_pred_proba_custom = model_custom.predict(X)
    logloss_custom = log_loss(y, y_pred_proba_custom)
    
    y_pred_proba_reference = reference_model.predict_proba(X)[:, 1]
    logloss_reference = log_loss(y, y_pred_proba_reference)

    x1_mesh, x2_mesh, x_mesh = create_mesh_grid(X)
    y_pred_mesh_custom = model_custom.classify(x_mesh).reshape(x1_mesh.shape)
    y_pred_mesh_reference = reference_model.predict(x_mesh).reshape(x1_mesh.shape)
       
    custom_cmap = ListedColormap(list(cmap_colors))

    custom_legend = [
        plt.Line2D(
            [0], [0], color=ref_color, linestyle='-', alpha=0.4, 
            linewidth=4, label=f'Reference boundary, NLL: {logloss_reference:.2f}'
        ),
        plt.Line2D(
            [0], [0], color=custom_color, linestyle='--', alpha=1, 
            linewidth=2, label=f'Custom boundary, NLL: {logloss_custom:.2f}'
        ),
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor=cmap_colors[0], 
            markersize=10, label='Class 0'
        ),
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor=cmap_colors[1], 
            markersize=10, label='Class 1'
        )
    ]

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=custom_cmap, label='Data')
    plt.contour(
        x1_mesh, x2_mesh, y_pred_mesh_reference, levels=[0.5], colors=ref_color, 
        linewidths=4, alpha=0.4, linestyles='-'
    )
    plt.contour(
        x1_mesh, x2_mesh, y_pred_mesh_custom, levels=[0.5], colors=custom_color, 
        alpha=1, linestyles='--'
    )
    plt.legend(handles=custom_legend)
    plt.xlabel('Feature $x_1$')
    plt.ylabel('Feature $x_2$')
    plt.title('Comparison of logistic regression implementations')
    plt.show()
