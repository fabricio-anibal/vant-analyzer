resultsRandomDir = "../vant-simulator/results/random"
resultsCentroideDir = "../vant-simulator/results/centroide"

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from mpl_toolkits.mplot3d import Axes3D

def listar_arquivos(diretorio):
    arquivos = []
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            arquivos.append(os.path.join(root, file))
    return arquivos

def cria_grafo():
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

class Graph:
    def _init_(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def draw(self):
        # Criar a figura 3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Desenhar nós
        for node, (x, y, z) in self.nodes:
            ax.scatter(x, y, z, color='lightblue', s=200)
            ax.text(x, y, z, str(node), color='black', fontsize=12, ha='center')

        # Desenhar arestas
        nodes_dict = {node: (x, y, z) for node, (x, y, z) in self.nodes}

        for edge in self.edges:
            x_vals = [nodes_dict[edge[0]][0], nodes_dict[edge[1]][0]]
            y_vals = [nodes_dict[edge[0]][1], nodes_dict[edge[1]][1]]
            z_vals = [nodes_dict[edge[0]][2], nodes_dict[edge[1]][2]]
            ax.plot(x_vals, y_vals, z_vals, color='gray')

        # Configuração dos eixos
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

def plot(values, title, filename):
    # Sort medias by valueMed (first element of the tuple)
    values.sort(key=lambda x: float(x[0]))
    value, num_nodes = zip(*values)
    value = [float(v) for v in value]  # Convert elements to floats
    plt.figure(figsize=(10, 6))
    plt.bar(num_nodes, value, color='b', alpha=0.6, label='Bar')
    plt.plot(num_nodes, value, color='r', marker='o', label='Line')

    # Annotate the points with exact values
    for i, v in enumerate(value):
        plt.text(num_nodes[i], v + 0.03 * max(value), f'({num_nodes[i]}, {v:.2f})', ha='center', va='bottom', fontsize=7, color='black')

    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Value (valueMed)')
    plt.title('Average Values vs Number of Nodes')
    plt.ylim(0, max(value) * 1.1)  # Ensure y-axis starts at 0 and add some padding on top
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join("./results", filename))
    plt.show()

def processRandomResults():
    randomResults = listar_arquivos(resultsRandomDir)

    medias = []
    minimos = []
    maximos = []

    for randomResult in randomResults:
        with open(randomResult, 'r') as file:
            lines = file.readlines()
            edges = []
            nodes = []
            node = None
            for line in lines:
                if line.startswith("Node"):
                    match = re.search(r'Node (\d+) \(([\d.]+), ([\d.]+), ([\d.]+)\)', line)
                    if match:
                        node = match.group(1)
                        x = float(match.group(2))
                        y = float(match.group(3))
                        z = float(match.group(4))
                        nodes.append((node, (x, y, z)))
                elif line.startswith("  -> Node "):
                    match = re.search(r'  -> Node (\d+), Weight', line)
                    if match:
                        nodeDestino = match.group(1)
                        edges.append((node, nodeDestino))
                elif line.startswith("Minimo: "):
                    match = re.search(r'Minimo: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        minimo = match.group(1)
                        valueMin = match.group(2)
                        minimos.append((valueMin, len(nodes)))
                elif line.startswith("Maximo: "):
                    match = re.search(r'Maximo: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        maximo = match.group(1)
                        valueMax = match.group(2)
                        maximos.append((valueMax, len(nodes)))
                elif line.startswith("Media: "):
                    match = re.search(r'Media: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        media = match.group(1)
                        valueMed = match.group(2)
                        medias.append((valueMed, len(nodes)))

            #graph = Graph(nodes, edges)
            #graph.draw()
    #plot(medias)
    plot(medias, "Average Values vs Number of Nodes for the Random Algorithm", "average_values_vs_number_of_nodes_random_algorithm.png")
    plot(minimos, "Minimum Values vs Number of Nodes for the Random Algorithm", "minimum_values_vs_number_of_nodes_random_algorithm.png")
    plot(maximos, "Maximum Values vs Number of Nodes for the Random Algorithm", "maximum_values_vs_number_of_nodes_random_algorithm.png")

def processCentroideResults():
    centroideResults = listar_arquivos(resultsCentroideDir)

    medias = []
    minimos = []
    maximos = []

    for centroideResult in centroideResults:
        with open(centroideResult, 'r') as file:
            lines = file.readlines()
            edges = []
            nodes = []
            node = None
            for line in lines:
                if line.startswith("Node"):
                    match = re.search(r'Node (\d+) \(([\d.]+), ([\d.]+), ([\d.]+)\)', line)
                    if match:
                        node = match.group(1)
                        x = float(match.group(2))
                        y = float(match.group(3))
                        z = float(match.group(4))
                        nodes.append((node, (x, y, z)))
                elif line.startswith("  -> Node "):
                    match = re.search(r'  -> Node (\d+), Weight', line)
                    if match:
                        nodeDestino = match.group(1)
                        edges.append((node, nodeDestino))
                elif line.startswith("Minimo: "):
                    match = re.search(r'Minimo: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        minimo = match.group(1)
                        valueMin = match.group(2)
                        minimos.append((valueMin, len(nodes)))
                elif line.startswith("Maximo: "):
                    match = re.search(r'Maximo: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        maximo = match.group(1)
                        valueMax = match.group(2)
                        maximos.append((valueMax, len(nodes)))
                elif line.startswith("Media: "):
                    match = re.search(r'Media: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        media = match.group(1)
                        valueMed = match.group(2)
                        medias.append((valueMed, len(nodes)))

            #graph = Graph(nodes, edges)
            #graph.draw()
    plot(medias, "Average Values vs Number of Nodes for the Centroid Algorithm", "average_values_vs_number_of_nodes_centroid_algorithm.png")
    plot(minimos, "Minimum Values vs Number of Nodes for the Centroid Algorithm", "minimum_values_vs_number_of_nodes_centroid_algorithm.png")
    plot(maximos, "Maximum Values vs Number of Nodes for the Centroid Algorithm", "maximum_values_vs_number_of_nodes_centroid_algorithm.png")

processCentroideResults()
processRandomResults()