import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph_from_matrices(matrix_A, matrix_B):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Number of nodes (assuming square matrices)
    num_nodes = len(matrix_A)
    
    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))
    
    # Add edges based on matrix A
    for i in range(num_nodes):
        for j in range(num_nodes):
            if matrix_A[i][j]:
                if G.has_edge(i, j):
                    # If edge exists, set label to 'A+B'
                    G[i][j]['label'] = 'A+B'
                else:
                    # Add edge with label 'A'
                    G.add_edge(i, j, color='red', label='A')  # Red for matrix A transitions

    # Add edges based on matrix B
    for i in range(num_nodes):
        for j in range(num_nodes):
            if matrix_B[i][j]:
                if G.has_edge(i, j):
                    # If edge exists, set label to 'A+B'
                    G[i][j]['label'] = 'A+B'
                else:
                    # Add edge with label 'B'
                    G.add_edge(i, j, color='blue', label='B')  # Blue for matrix B transitions

    # Draw the graph
    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    labels = nx.get_edge_attributes(G, 'label')
    
    pos = nx.spring_layout(G)  # Position the nodes using the spring layout
    nx.draw(G, pos, with_labels=True, node_color='lightgrey', edge_color=colors, node_size=500, font_size=16, font_weight='bold', arrows=True)
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='black', font_size=12)
    
    plt.title('Graph Visualization')
    plt.show()

# Define matrices A and B
matrix_A = [[False, True, False],
 [False, True, False],
 [True, False, False]]

matrix_B = [[True, False, False],
 [False, False, True],
 [False, False, True]]

# Visualize the graph
visualize_graph_from_matrices(matrix_A, matrix_B)
