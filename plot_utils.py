import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from constants import REACTOR_BLOCK_ID, REACTOR_BLOCK_COLOR

def plot_grid(array_3d, save_path=None):
    num_layers = array_3d.shape[0]  # Number of layers
    
    # Determine the figure size based on array dimensions
    layer_size = array_3d.shape[1:]  # Only need X and Y dimensions
    figsize = (layer_size[1] / 2, (layer_size[0] / 2) * num_layers)

    fig, axes = plt.subplots(nrows=num_layers, ncols=1, figsize=figsize)

    # Ensure axes is always iterable
    if num_layers == 1:
        axes = [axes]

    # Plot each layer as a separate subplot
    for i in range(num_layers):
        layer_data = array_3d[i]
        ax = axes[i]
        ax.set_title(f'Layer {i + 1}')

        # Create a custom colormap based on block_colors
        cmap = ListedColormap([REACTOR_BLOCK_COLOR.get(key, 'white') for key in range(-1, 17)])

        # Plot the layer with specific colors and labels
        im = ax.imshow(layer_data, cmap=cmap, vmin=-1, vmax=16, aspect='equal')

        # Add text annotations for each cell
        for y in range(layer_data.shape[0]):
            for x in range(layer_data.shape[1]):
                value = layer_data[y, x]
                label = REACTOR_BLOCK_ID.get(value, str(value))
                ax.text(x, y, label, ha='center', va='center', color='black', fontsize=10)

        # Add colorbar
        #cbar = fig.colorbar(im, ax=ax, ticks=np.arange(-1, 17))
        #cbar.ax.set_yticklabels([REACTOR_BLOCK_ID.get(j, str(j)) for j in range(-1, 17)])
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()