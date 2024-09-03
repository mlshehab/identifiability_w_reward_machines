import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def plot_arrows(matrix, values, u,gamma):
    n = matrix.shape[0]
    
    # Create a grid of points, offset by 0.5 to center the arrows
    X, Y = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    
    # Initialize direction arrays
    U = np.zeros((n, n))
    V = np.zeros((n, n))
    
    # Set directions based on the matrix values
    V[matrix == 0] = 1  # Down
    U[matrix == 1] = 1   # Right
    V[matrix == 2] = -1   # Up
    U[matrix == 3] = -1  # Left
    
    # Create the plot with a scale to reduce arrow size
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Add a green patch covering the 4 grid cells in the bottom left corner
    rect1 = patches.Rectangle((0, n-2), 2, 2, linewidth=1, edgecolor='none', facecolor='green', alpha=0.3)
    rect2 = patches.Rectangle((n-2, 0), 2, 2, linewidth=1, edgecolor='none', facecolor='red', alpha=0.3)
    rect3 = patches.Rectangle((n-2, n-2), 2, 2, linewidth=1, edgecolor='none', facecolor='yellow', alpha=0.3)
    rect4 = patches.Rectangle((0, 0), 2, 2, linewidth=1, edgecolor='none', facecolor='blue', alpha=0.3)
   
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)

    # Plot the arrows
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=4)  # Increased scale value reduces arrow size
    
    # Add the values from the second matrix under each arrow
    for i in range(n):
        for j in range(n):
            ax.text(j + 0.5, i + 0.5 + 0.35, f"{values[i, j]:.2f}", ha='center', va='center', fontsize=12, color='black')
    
    
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()  # Invert Y-axis to match matrix indexing
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, n+1))
    ax.set_yticks(np.arange(0, n+1))
    ax.grid(True)
    ax.set_title(f"Policy for RM state $u = {u}$ and $\gamma= {gamma}$")
    plt.show()


