resultsRandomDir = "../../vant-simulator/results/random"
resultsCentroideDir = "../../vant-simulator/centroide/random"

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Criar um grafo
G = nx.Graph()

# Adicionar nós com coordenadas 3D
nodes = {i: (np.random.rand(), np.random.rand(), np.random.rand()) for i in range(1, 6)}
G.add_nodes_from(nodes.keys())

# Adicionar arestas aleatórias
edges = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
G.add_edges_from(edges)

# Criar a figura 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Desenhar nós
for node, (x, y, z) in nodes.items():
    ax.scatter(x, y, z, color='lightblue', s=200)
    ax.text(x, y, z, str(node), color='black', fontsize=12, ha='center')

# Desenhar arestas
for edge in edges:
    x_vals = [nodes[edge[0]][0], nodes[edge[1]][0]]
    y_vals = [nodes[edge[0]][1], nodes[edge[1]][1]]
    z_vals = [nodes[edge[0]][2], nodes[edge[1]][2]]
    ax.plot(x_vals, y_vals, z_vals, color='gray')

# Configuração dos eixos
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
