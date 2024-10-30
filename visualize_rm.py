import matplotlib.pyplot as plt
import networkx as nx

# Define the matrices and propositions
matrices =[[False, False, False, True],
 [False, True, False, False],
 [False, True, False, False],
 [False, False, False, True]],
[[True, False, False, False],
 [False, True, False, False],
 [False, False, True, False],
 [False, True, False, False]],
[[True, False, False, False],
 [False, False, True, False],
 [False, False, True, False],
 [False, False, False, True]],
[[True, False, False, False],
 [False, True, False, False],
 [False, False, True, False],
 [False, False, False, True]],
[[True, False, False, False],
 [False, True, False, False],
 [False, False, True, False],
 [False, False, False, True]]


propositions = ['A', 'B', 'C', 'D', 'H']

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
                # If the edge already exists, we append the label to avoid overwriting it
                if G.has_edge(i + 1, j + 1):
                    G.edges[i + 1, j + 1]['label'] += f', {prop}'
                else:
                    G.add_edge(i + 1, j + 1, label=prop)  # Add the edge with the proposition as label

# Draw the graph
pos = nx.spring_layout(G)  # Position the nodes

# Draw nodes and edges (no curvature)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold')

# Draw edge labels, including self-loops
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red')

# Show the plot
plt.title("Mealy Machine Graph with Straight Edges")
plt.show()