import matplotlib.pyplot as plt
import networkx as nx
import ast
import math

# Step 1: Read the matrix solutions from a file
import ast

def read_solutions(filename):
    solutions = []
    current_solution = []
    current_matrix = ""  # To accumulate matrix lines
    
    with open(filename, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            
            if stripped_line.startswith("Solution"):
                if current_solution:  # Append the previous solution
                    solutions.append(current_solution)
                current_solution = []  # Start a new solution
                current_matrix = ""  # Reset the current matrix accumulator
            
            elif stripped_line:  # Non-empty line
                # Accumulate matrix rows
                current_matrix += stripped_line
                
                # Check if the matrix is complete (ends with ']]')
                if stripped_line.endswith(']]'):
                    current_solution.append(ast.literal_eval(current_matrix))  # Parse the matrix
                    current_matrix = ""  # Reset for next matrix
    
    # Append the last solution if any
    if current_solution:
        solutions.append(current_solution)

    return solutions



# Step 2: Define the function to draw the graph from a matrix
def draw_graph(matrices, ax, title):
    propositions = ['A', 'B', 'C', 'D']

    # Number of nodes
    n = len(matrices[0])

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges based on the matrices
    for idx, matrix in enumerate(matrices):
        prop = propositions[idx]  # Proposition for the current matrix
        for i in range(n):
            for j in range(n):
                if matrix[i][j]:  # If there's a transition from node i to node j
                    if G.has_edge(i + 1, j + 1):
                        G.edges[i + 1, j + 1]['label'] += f', {prop}'
                    else:
                        G.add_edge(i + 1, j + 1, label=prop)  # Add the edge with the proposition as label

    # Draw the graph
    pos = nx.spring_layout(G)  # Position the nodes

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=12, font_weight='bold', ax = ax)
    edge_labels = nx.get_edge_attributes(G, 'label')

    # Draw labels for straight edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red',ax = ax)
    ax.set_title(title, fontsize=10)
# Step 3: Create subplots for each solution
def plot_solutions(solutions):
    num_solutions = len(solutions)
    grid_size = math.ceil(math.sqrt(num_solutions))  # Determine grid size for subplots

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()  # Flatten to easily index each subplot

    for idx, solution in enumerate(solutions):
        draw_graph(solution, axes[idx], f"Solution {idx}")

    # Remove extra empty subplots
    for ax in axes[num_solutions:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Step 4: Main function to run the script
def main():
    filename = 'solutions.txt'  # Replace with your file path
    solutions = read_solutions(filename)
    plot_solutions(solutions)

if __name__ == "__main__":
    main()
