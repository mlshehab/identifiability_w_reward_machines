import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Define the matrices and propositions
matrices = [
    [[False, False, True, False],
     [False, True, False, False],
     [False, False, True, False],
     [False, True, False, False]],
    [[True, False, False, False],
     [False, True, False, False],
     [False, True, False, False],
     [False, False, False, True]],
    [[False, False, True, False],
     [False, False, False, True],
     [False, False, True, False],
     [False, False, False, True]],
    [[True, False, False, False],
     [False, True, False, False],
     [False, True, False, False],
     [True, False, False, False]]
]

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
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', ax = ax)

# Edge and label handling
arc_rad = 0.3  # Radius for curvature of the edges
straight_edges = []
curved_edges = []

for (u, v, d) in G.edges(data=True):
    if G.has_edge(v, u) and u != v:  # If there's an edge in the opposite direction, curve both
        if (v, u) not in curved_edges:  # Avoid duplicate curved edges
            curved_edges.append((u, v))
    else:
        straight_edges.append((u, v))

# Draw straight edges with arrow tips
# nx.draw_networkx_edges(G, pos, edgelist=straight_edges, connectionstyle='arc3,rad=0', edge_color='black', arrows=True)

# Draw curved edges with arrow tips
# nx.draw_networkx_edges(G, pos, edgelist=curved_edges, connectionstyle=f'arc3,rad={arc_rad}', edge_color='black', arrows=True, arrowstyle='->')

# Get edge labels
edge_labels = nx.get_edge_attributes(G, 'label')

# Draw labels for straight edges
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red')

# Manually draw labels for curved edges
for (u, v) in curved_edges:
    # Calculate mid-point for the curved edge label and offset it
    label_pos = (pos[u] + pos[v]) / 2  # Mid-point
    # Offset the label to position it closer to the curve
    offset = np.array([0.1, 0.1]) * (pos[u] - pos[v]) / np.linalg.norm(pos[u] - pos[v])
    label_pos += offset
    label = G[u][v]['label']
    plt.text(label_pos[0], label_pos[1], label, fontsize=10, color='red', horizontalalignment='center')

# Show the plot
plt.title("Graph with Curved Edges, Arrow Tips, and Labels Near Curves")
plt.show()
