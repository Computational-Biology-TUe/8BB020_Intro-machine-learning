import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np


def draw_network(ax, edges, labels, edge_labels, pos, title, include_edge_labels=True):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    square_keywords = ['x', 'y', 'h', 'bias']
    square_nodes = [node for node, label in labels.items() if any(keyword in label for keyword in square_keywords)]
    circle_nodes = [node for node in G.nodes() if node not in square_nodes]
    bias_edges = [(u, v) for u, v in G.edges() if 'bias' in labels.get(u, '')]

    node_size_circle = 2000
    node_size_square = 1400
    
    node_colors = {node: 'lightblue' if node in circle_nodes else 'lightcoral' for node in G.nodes()}
    
    nx.draw_networkx_nodes(G, pos, nodelist=circle_nodes, node_shape='o', node_size=node_size_circle, node_color=[node_colors[node] for node in circle_nodes], ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=square_nodes, node_shape='s', node_size=node_size_square, node_color=[node_colors[node] for node in square_nodes], ax=ax)
    
    nx.draw_networkx_edges(G, pos, edgelist=[e for e in G.edges() if e not in bias_edges], ax=ax, arrows=True, arrowsize=20, edge_color='black', node_size=node_size_circle, connectionstyle='arc3,rad=0.01')
    nx.draw_networkx_edges(G, pos, edgelist=bias_edges, ax=ax, arrows=True, arrowsize=20, edge_color='black', node_size=node_size_circle, connectionstyle='arc3,rad=0.15')
    
    if include_edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=12, label_pos=0.3)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black', ax=ax)
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
        
    legend_elements = [
        Patch(facecolor='lightcoral', edgecolor='black', label='Variables'),
        Patch(facecolor='lightblue', edgecolor='black', label='Computation'),
        Line2D([0], [0], color='black', lw=2, label='Connection')
    ]
    ax.legend(handles=legend_elements, loc='upper right')


def plot_linear_regression(include_bias=True):
    fig, ax = plt.subplots()
    edges = [(1, 3), (2, 3), (3, 4)]
    labels = {1: r'$x_1$', 2: r'$x_2$', 3: r'$\Sigma$', 4: r'$y$'}
    edge_labels = {(1, 3): r'$\theta_1$', (2, 3): r'$\theta_2$'}
    pos = {1: (0, 0.5), 2: (0, -0.5), 3: (1, 0), 4: (2, 0)}
    
    if include_bias:
        bias_node = 5
        edges.append((bias_node, 3))
        labels[bias_node] = '1 \n(bias)'
        edge_labels[(bias_node, 3)] = r'$\theta_0$'
        pos[bias_node] = (0.5, -1)
    
    draw_network(ax, edges, labels, edge_labels, pos, "Linear Regression")
    plt.show()


def plot_logistic_regression(include_bias=True):
    fig, ax = plt.subplots()
    edges = [(1, 3), (2, 3), (3, 4), (4, 5)]
    labels = {1: r'$x_1$', 2: r'$x_2$', 3: r'$\Sigma$', 4: r'$\sigma$', 5: r'$p(y=1)$'}
    edge_labels = {(1, 3): r'$\theta_1$', (2, 3): r'$\theta_2$'}
    pos = {1: (0, 0.5), 2: (0, -0.5), 3: (1, 0), 4: (2, 0), 5: (3, 0)}
    
    if include_bias:
        bias_node = 6
        edges.append((bias_node, 3))
        labels[bias_node] = '1 \n(bias)'
        edge_labels[(bias_node, 3)] = r'$\theta_0$'
        pos[bias_node] = (0.5, -1)
    
    draw_network(ax, edges, labels, edge_labels, pos, "Logistic Regression")
    plt.show()



def plot_sigmoid_relu_tanh():
    # Generate 100 equally spaced values from -10 to 10
    x = np.linspace(-10, 10, 100)
    
    # Compute the sigmoid function values
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Compute ReLU values
    relu = np.maximum(0, x)
    
    # Compute tanh values
    tanh = np.tanh(x)
    
    # Create subplots: 1 row, 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot sigmoid on the first subplot
    axs[0].plot(x, sigmoid)
    axs[0].set_xlabel('Input')
    axs[0].set_ylabel('Output')
    axs[0].set_title('Sigmoid Activation Function')
    axs[0].grid(True)
    
    # Plot ReLU on the second subplot
    axs[1].plot(x, relu)
    axs[1].set_xlabel('Input')
    axs[1].set_ylabel('Output')
    axs[1].set_title('ReLU Activation Function')
    axs[1].grid(True)
    
    # Plot tanh on the third subplot
    axs[2].plot(x, tanh)
    axs[2].set_xlabel('Input')
    axs[2].set_ylabel('Output')
    axs[2].set_title('Tanh Activation Function')
    axs[2].grid(True)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Show the plot
    plt.show()

     

def plot_quadratic_logistic_regression(include_bias=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    edges = [
        (1, 7), (2, 8), (1, 9), (2, 9), (1, 4), (2, 4), 
        (7, 10), (8, 11), (9, 12), (10, 4), (11, 4), (12, 4), 
        (4, 5), (5, 6)
    ]
    labels = {
        1: r'$x_1$', 2: r'$x_2$', 4: r'$\Sigma$', 5: r'$\sigma$', 6: r'$p(y=1)$', 
        7: r'$**2$', 8: r'$**2$', 9: r'$*$', 10: r'$x_1^2$', 11: r'$x_2^2$', 12: r'$x_1 x_2$'
    }
    edge_labels = {
        (10, 4): r'$\theta_3$', (11, 4): r'$\theta_4$', (12, 4): r'$\theta_5$', 
        (1, 4): r'$\theta_1$', (2, 4): r'$\theta_2$'
    }
    pos = {
        1: (0, 1), 2: (0, -1), 7: (1, 2), 8: (1, -2), 9: (1, 0), 
        10: (2, 2), 11: (2, -2), 12: (2, 0), 4: (3, 0), 5: (5, 0), 6: (7, 0)
    }
    
    if include_bias:
        bias_node = 3
        edges.append((bias_node, 4))
        labels[bias_node] = '1 \n(bias)'
        edge_labels[(bias_node, 4)] = r'$\theta_0$'
        pos[bias_node] = (2.25, -3)
    
    draw_network(ax, edges, labels, edge_labels, pos, "Quadratic Logistic Regression")
    plt.show()


def takes_two_to_xor(include_bias=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    edges = [(1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 9), (9, 10), (10, 11)]
    labels = {1: r'$x_1$', 2: r'$x_2$', 3: r'$\Sigma$', 4: r'$\Sigma$', 5: r'$\sigma$', 6: r'$\sigma$', 7: r'$h_1$', 8: r'$h_2$', 9: r'$\Sigma$', 10: r'$\sigma$', 11: r'$p(y=1)$'}
    edge_labels = {(1, 3): r'$\theta_{11}$', (1, 4): r'$\theta_{12}$', 
                   (2, 3): r'$\theta_{21}$', (2, 4): r'$\theta_{22}$',                    
                   (7, 9): r'$\theta_{31}$', (8, 9): r'$\theta_{32}$'}
    pos = {1: (0, 1), 2: (0, -1), 3: (1, 0.5), 4: (1, -0.5), 5: (2, 0.5), 6: (2, -0.5), 7: (3, 0.5), 8: (3, -0.5), 9: (4, 0), 10: (5, 0), 11: (6, 0)}
    
    if include_bias:
        bias_node = 12
        edges.extend([(bias_node, 3), (bias_node, 4), (bias_node, 9)])
        labels[bias_node] = '1 \n(bias)'
        edge_labels[(bias_node, 3)] = r'$\theta_{01}$'
        edge_labels[(bias_node, 4)] = r'$\theta_{02}$'
        edge_labels[(bias_node, 9)] = r'$\theta_{03}$'
        pos[bias_node] = (2, -2)
        
    draw_network(ax, edges, labels, edge_labels, pos, "Combining the output of two Logistic Regression models")
    plt.show()
    

def small_nn(include_bias=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    edges = [(1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)]
    labels = {1: r'$x_1$', 2: r'$x_2$', 3: '', 4: '', 5: '', 6: r'$p(y=1)$'}
    edge_labels = {(1, 3): r'$\theta_{11}$', (1, 4): r'$\theta_{12}$', 
                   (2, 3): r'$\theta_{21}$', (2, 4): r'$\theta_{22}$',                    
                   (3, 5): r'$\theta_{31}$', (4, 5): r'$\theta_{32}$'}
    pos = {1: (0, 1), 2: (0, -1), 3: (1, 0.5), 4: (1, -0.5), 5: (2, 0), 6: (3, 0)}
    
    if include_bias:
        bias_node = 7
        edges.extend([(bias_node, 3), (bias_node, 4), (bias_node, 5)])
        labels[bias_node] = '1 \n(bias)'
        edge_labels[(bias_node, 3)] = r'$\theta_{01}$'
        edge_labels[(bias_node, 4)] = r'$\theta_{02}$'
        edge_labels[(bias_node, 5)] = r'$\theta_{03}$'
        pos[bias_node] = (1.25, -1.5)
        
    draw_network(ax, edges, labels, edge_labels, pos, "Simplified XOR Neural Network")
    plt.show()


def plot_neural_network(num_inputs, num_hidden_layers, neurons_per_layer, include_edge_labels=True):
    fig, ax = plt.subplots(figsize=(15, 10))
    edges = []
    labels = {}
    edge_labels = {}
    pos = {}

    if isinstance(neurons_per_layer, int):
        neurons_per_layer = (neurons_per_layer,) * num_hidden_layers

    max_neurons_per_layer = max(num_inputs, *neurons_per_layer)

    input_offset = (max_neurons_per_layer - num_inputs) / 2
    hidden_offsets = [(max_neurons_per_layer - n) / 2 for n in neurons_per_layer]
    output_offset = (max_neurons_per_layer - 1) / 2

    for i in range(num_inputs):
        labels[i + 1] = r'$x_{%d}$' % (i + 1)
        pos[i + 1] = (0, max_neurons_per_layer - input_offset - i - 1)

    node_idx = num_inputs + 1
    previous_layer_hidden = range(1, num_inputs + 1)
    
    for layer in range(num_hidden_layers):
        num_neurons = neurons_per_layer[layer]
        current_layer_sum = range(node_idx, node_idx + num_neurons)
        current_layer_sigma = range(node_idx + num_neurons, node_idx + 2 * num_neurons)
        current_layer_hidden = range(node_idx + 2 * num_neurons, node_idx + 3 * num_neurons)
        
        for i, prev_node in enumerate(previous_layer_hidden):
            for j, curr_node in enumerate(current_layer_sum):
                edges.append((prev_node, curr_node))
                edge_labels[(prev_node, curr_node)] = r'$\theta_{%d%d%d}$' % (layer + 1, i + 1, j + 1)

        for curr_node in current_layer_sum:
            labels[curr_node] = r'$\Sigma$'
            labels[curr_node + num_neurons] = r'$\sigma$'
            pos[curr_node] = (layer * 4 + 1, max_neurons_per_layer - hidden_offsets[layer] - (curr_node - node_idx) - 1)
            pos[curr_node + num_neurons] = (layer * 4 + 2, max_neurons_per_layer - hidden_offsets[layer] - (curr_node - node_idx) - 1)
            edges.append((curr_node, curr_node + num_neurons))
        
        for j, curr_node in enumerate(current_layer_sigma):
            hidden_node = node_idx + 2 * num_neurons + j
            labels[hidden_node] = r'$h_{%d%d}$' % (layer + 1, j + 1)
            pos[hidden_node] = (layer * 4 + 3, max_neurons_per_layer - hidden_offsets[layer] - j - 1)
            edges.append((curr_node, hidden_node))
        
        node_idx += 3 * num_neurons
        previous_layer_hidden = current_layer_hidden

    output_sum = node_idx
    output_sigma = node_idx + 1
    labels[output_sum] = r'$\Sigma$'
    labels[output_sigma] = r'$\sigma$'
    labels[output_sigma + 1] = r'$p(y=1)$'
    pos[output_sum] = (num_hidden_layers * 4 + 1, max_neurons_per_layer - output_offset - 1)
    pos[output_sigma] = (num_hidden_layers * 4 + 2, max_neurons_per_layer - output_offset - 1)
    pos[output_sigma + 1] = (num_hidden_layers * 4 + 3, max_neurons_per_layer - output_offset - 1)
    
    for i, prev_node in enumerate(previous_layer_hidden):
        edges.append((prev_node, output_sum))
        edge_labels[(prev_node, output_sum)] = r'$\theta_{%d%d}$' % (num_hidden_layers + 1, i + 1)
    
    edges.append((output_sum, output_sigma))
    edges.append((output_sigma, output_sigma + 1))

    draw_network(ax, edges, labels, edge_labels, pos, "Neural Network", include_edge_labels)
    plt.show()
