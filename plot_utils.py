import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from constants import REACTOR_BLOCK_ID, REACTOR_BLOCK_COLOR

def plot_grid(array_3d, save_path=None):
    num_layers = array_3d.shape[0]  # Number of layers
    fig, axes = plt.subplots(nrows=num_layers, ncols=1, figsize=(5, 5*num_layers))

    # Plot each layer as a separate subplot
    for i in range(num_layers):
        layer_data = array_3d[i]
        ax = axes[i]
        ax.set_title(f'Layer {i+1}')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Create a custom colormap based on block_colors
        cmap = ListedColormap([REACTOR_BLOCK_COLOR.get(key, 'white') for key in range(18)])

        # Plot the layer with specific colors
        im = ax.imshow(layer_data, cmap=cmap, vmin=0, vmax=17, aspect='equal')

        # Set ticks and grid to match block boundaries
        ax.set_xticks(np.arange(layer_data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(layer_data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, 18))
        cbar.ax.set_yticklabels([REACTOR_BLOCK_ID.get(j, str(j)) for j in range(18)])

    plt.tight_layout()

    # Save or show plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()