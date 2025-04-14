import shutil

resultsRandomDir = "../vant-simulator/results/random"
resultsCentroideDir = "../vant-simulator/results/centroide"
dirPrefix = "../vant-simulator"

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict

def listar_arquivos(diretorio):
    arquivos = []
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            arquivos.append(os.path.join(root, file))
    return arquivos

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.remove_duplicated_edges()

    def remove_duplicated_edges(self):
        unique_edges = set()
        for edge in self.edges:
            if edge[0] != edge[1]:
                unique_edges.add(tuple(sorted(edge)))
        self.edges = list(unique_edges)

    def draw(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for node, (x, y, z) in self.nodes:
            ax.scatter(x, y, z, color='lightblue', s=200)
            ax.text(x, y, z, str(node), color='black', fontsize=12, ha='center')

        nodes_dict = {node: (x, y, z) for node, (x, y, z) in self.nodes}

        for edge in self.edges:
            x_vals = [nodes_dict[edge[0]][0], nodes_dict[edge[1]][0]]
            y_vals = [nodes_dict[edge[0]][1], nodes_dict[edge[1]][1]]
            z_vals = [nodes_dict[edge[0]][2], nodes_dict[edge[1]][2]]
            ax.plot(x_vals, y_vals, z_vals, color='gray')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def calculate_density(self):
        num_edges = len(self.edges)
        num_nodes = len(self.nodes)
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_possible_edges
        return density

    def calculate_density_without_node(self, node):
        filtered_edges = [edge for edge in self.edges if node[0] not in edge]
        filtered_nodes = [n for n in self.nodes if n != node]

        num_edges = len(filtered_edges)
        num_nodes = len(filtered_nodes)

        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_possible_edges
        return density



def plot(pasta, values1, values2, title, filename, values1Lable, values2Lable):
    # Sort values1 and values2 by num_nodes (second element of the tuple)
    values1.sort(key=lambda x: int(x[1]))
    values2.sort(key=lambda x: int(x[1]))

    value1, num_nodes1 = zip(*values1)
    value2, num_nodes2 = zip(*values2)

    value1 = [float(v) for v in value1]  # Convert elements to floats
    value2 = [float(v) for v in value2]  # Convert elements to floats

    plt.figure(figsize=(10, 6))

    # Plot values1
    plt.plot(num_nodes1, value1, color='r', marker='o', label=values1Lable)
    for i, (v1, n1) in enumerate(zip(value1, num_nodes1)):
        # Verifica o valor correspondente em values2 para o mesmo num_nodes
        idx2 = num_nodes2.index(n1)
        v2 = value2[idx2]

        # Determina se o valor de value1 ou value2 é maior e posiciona o rótulo
        if v1 > v2:
            plt.text(n1, v1 + 0.1 * max(value1), f'({n1}, {v1:.2f})', ha='center', va='bottom', fontsize=7, color='black')
            plt.text(n1, v2 - 0.1 * max(value2), f'({n1}, {v2:.2f})', ha='center', va='top', fontsize=7, color='black')
        else:
            plt.text(n1, v2 + 0.1 * max(value2), f'({n1}, {v2:.2f})', ha='center', va='bottom', fontsize=7, color='black')
            plt.text(n1, v1 - 0.1 * max(value1), f'({n1}, {v1:.2f})', ha='center', va='top', fontsize=7, color='black')

    # Plot values2
    plt.plot(num_nodes2, value2, color='b', marker='o', label=values2Lable)
    for i, (v2, n2) in enumerate(zip(value2, num_nodes2)):
        # Verifica o valor correspondente em values1 para o mesmo num_nodes
        idx1 = num_nodes1.index(n2)
        v1 = value1[idx1]

        # Determina se o valor de value1 ou value2 é maior e posiciona o rótulo
        if v2 > v1:
            plt.text(n2, v2 + 0.1 * max(value2), f'({n2}, {v2:.2f})', ha='center', va='bottom', fontsize=7, color='black')
            plt.text(n2, v1 - 0.1 * max(value1), f'({n2}, {v1:.2f})', ha='center', va='top', fontsize=7, color='black')
        else:
            plt.text(n2, v1 + 0.1 * max(value1), f'({n2}, {v1:.2f})', ha='center', va='bottom', fontsize=7, color='black')
            plt.text(n2, v2 - 0.1 * max(value2), f'({n2}, {v2:.2f})', ha='center', va='top', fontsize=7, color='black')

    plt.xlabel('Número de nós')
    plt.ylabel('Tempo médio das variações')
    plt.title(title)
    plt.ylim(0, max(max(value1), max(value2)) * 1.1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()

    plt.savefig(os.path.join(pasta, filename))
    plt.close()
    # plt.show()

def plotWithoutLable(pasta, values1, values2, title, filename, values1Lable, values2Lable):
    # Sort values1 and values2 by num_nodes (second element of the tuple)
    values1.sort(key=lambda x: int(x[1]))
    values2.sort(key=lambda x: int(x[1]))

    value1, num_nodes1 = zip(*values1)
    value2, num_nodes2 = zip(*values2)

    value1 = [float(v) for v in value1]
    value2 = [float(v) for v in value2]

    plt.figure(figsize=(10, 6))

    # Plot values1
    plt.plot(num_nodes1, value1, color='r', marker='o', label=values1Lable)
    #for v1, n1 in zip(value1, num_nodes1):
    #    idx2 = num_nodes2.index(n1)
    #    v2 = value2[idx2]
    #    if v1 > v2:
    #        plt.text(n1, v1 + 0.05 * (max(value1) - min(value1)), f'({n1}, {v1:.2f})',
    #                 ha='center', va='bottom', fontsize=7, color='black')
    #        plt.text(n1, v2 - 0.05 * (max(value2) - min(value2)), f'({n1}, {v2:.2f})',
    #                 ha='center', va='top', fontsize=7, color='black')
    #    else:
    #        plt.text(n1, v2 + 0.05 * (max(value2) - min(value2)), f'({n1}, {v2:.2f})',
    #                 ha='center', va='bottom', fontsize=7, color='black')
    #        plt.text(n1, v1 - 0.05 * (max(value1) - min(value1)), f'({n1}, {v1:.2f})',
    #                 ha='center', va='top', fontsize=7, color='black')

    # Plot values2
    plt.plot(num_nodes2, value2, color='b', marker='o', label=values2Lable)

    # Define limites com base nos valores mínimos e máximos com margens
    all_values = value1 + value2
    min_val = min(all_values)
    max_val = max(all_values)
    margin = 0.05 * (max_val - min_val if max_val > min_val else 1)

    plt.ylim(min_val - margin, max_val + margin)

    plt.xlabel('Número de nós')
    plt.ylabel('Taxa de transmissão média (bytes/segundo)')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()

    plt.savefig(os.path.join(pasta, filename))
    plt.close()

def agrupar_e_calcular_media(dados):
    grupos = defaultdict(list)
    for qtd_drones, densidade in dados:
        grupos[qtd_drones].append(densidade)

    # Retorna lista de tuplas: (quantidade_drones, média_densidade)
    return [(qtd, sum(densidades) / len(densidades)) for qtd, densidades in sorted(grupos.items())]


def plot_density_comparison(pasta, current, previous, filename, title):
    # Agrupa e calcula média
    current_avg = agrupar_e_calcular_media(current)
    previous_avg = agrupar_e_calcular_media(previous)

    x_labels = [x[0] for x in current_avg]
    y_current = [x[1] for x in current_avg]
    y_previous = [x[1] for x in previous_avg]

    x = np.arange(len(x_labels))
    width = 0.45

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_prev = ax.bar(x - width/2, y_previous, width, label='Aleatório', color='skyblue')
    bars_curr = ax.bar(x + width/2, y_current, width, label='Centróide' ,color='red')

    ax.set_xlabel('Quantidade de Drones')
    ax.set_ylabel('Densidade Média')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    minimo = min(min(y_previous), min(y_current))
    ax.set_ylim(minimo - 0.1, 1.1)

    # Fonte adaptável
    n_barras = len(x_labels)
    fontsize = max(6, 12 - n_barras // 5)

    def add_vertical_labels(bars1, bars2):
        for bar1, bar2 in zip(bars1, bars2):
            h1 = bar1.get_height()
            h2 = bar2.get_height()
            offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01  # Deslocamento base

            # Adiciona os valores das barras em vertical (sempre em cima das barras)
            ax.text(bar1.get_x() + bar1.get_width() / 2,
                    h1 + offset,
                    f'{h1:.3f}',
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=fontsize)

            ax.text(bar2.get_x() + bar2.get_width() / 2,
                    h2 + offset,
                    f'{h2:.3f}',
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=fontsize)

    add_vertical_labels(bars_prev, bars_curr)

    plt.tight_layout()
    plt.savefig(os.path.join(pasta, filename))
    plt.close()
    # plt.show()

def consolidateResults(values):
    values_zipped, num_nodes = zip(*values)

    map_values = {}

    for i, v in enumerate(num_nodes):
        if v not in map_values:
            map_values[v] = []
        map_values[v].append(values_zipped[i])
    
    #print(map_values)

    final = []

    for key in map_values:
        values_map = map_values[key]
        values_map = [float(value) for value in values_map]
        #print(values_map)
        media = sum(values_map) / len(values_map)
        final.append((media, key))
    
    return final

def processRandomResults(pasta):
    randomResults = listar_arquivos(dirPrefix + "/" + pasta + "/random")

    medias = []
    minimos = []
    maximos = []
    densidade_anterior = []
    densidade = []
    transmition = []

    for randomResult in randomResults:
        with open(randomResult, 'r') as file:
            lines = file.readlines()
            edges = []
            nodes = []
            transmitionAux = []
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
                    match = re.search(r'  -> Node (\d+), Weight: ([\d.]+), TransmitionRate: ([\d.]+)', line)
                    if match:
                        nodeDestino = match.group(1)
                        edges.append((node, nodeDestino))
                        peso = match.group(3)
                        transmitionAux.append(peso)
                elif line.startswith("Minimo: "):
                    match = re.search(r'Minimo: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        valueMin = match.group(2)
                        minimos.append((valueMin, len(nodes)))
                elif line.startswith("Maximo: "):
                    match = re.search(r'Maximo: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        valueMax = match.group(2)
                        maximos.append((valueMax, len(nodes)))
                elif line.startswith("Media: "):
                    match = re.search(r'Media: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        valueMed = match.group(2)
                        medias.append((valueMed, len(nodes)))

            graph = Graph(nodes, edges)
            densidade_anterior.append((len(nodes), graph.calculate_density_without_node(max(nodes))))
            densidade.append((len(nodes), graph.calculate_density()))
            pesoMedio = sum([float(peso) for peso in transmitionAux]) / len(transmitionAux)
            transmition.append((pesoMedio, len(nodes)))

    medias = consolidateResults(medias)
    minimos = consolidateResults(minimos)
    maximos = consolidateResults(maximos)
    transmition = consolidateResults(transmition)

    return {
        'medias': medias,
        'minimos': minimos,
        'maximos': maximos,
        'densidade': densidade,
        'densidade_anterior' : densidade_anterior,
        'transmition': transmition
    }

def processCentroideResults(pasta):
    centroideResults = listar_arquivos(dirPrefix + "/" + pasta + "/centroide")

    medias = []
    minimos = []
    maximos = []
    densidade_anterior = []
    densidade = []
    transmition = []

    for centroideResult in centroideResults:
        with open(centroideResult, 'r') as file:
            lines = file.readlines()
            edges = []
            nodes = []
            transmitionAux = []
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
                    match = re.search(r'  -> Node (\d+), Weight: ([\d.]+), TransmitionRate: ([\d.]+)', line)
                    if match:
                        nodeDestino = match.group(1)
                        edges.append((node, nodeDestino))
                        peso = match.group(3)
                        transmitionAux.append(peso)
                elif line.startswith("Minimo: "):
                    match = re.search(r'Minimo: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        valueMin = match.group(2)
                        minimos.append((valueMin, len(nodes)))
                elif line.startswith("Maximo: "):
                    match = re.search(r'Maximo: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        valueMax = match.group(2)
                        maximos.append((valueMax, len(nodes)))
                elif line.startswith("Media: "):
                    match = re.search(r'Media: \[(\d+)\] ([\d.]+)', line)
                    if match:
                        valueMed = match.group(2)
                        medias.append((valueMed, len(nodes)))

            graph = Graph(nodes, edges)
            densidade_anterior.append((len(nodes), graph.calculate_density_without_node(max(nodes))))
            densidade.append((len(nodes), graph.calculate_density()))
            pesoMedio = sum([float(peso) for peso in transmitionAux]) / len(transmitionAux)
            transmition.append((pesoMedio, len(nodes)))
    
    medias = consolidateResults(medias)
    minimos = consolidateResults(minimos)
    maximos = consolidateResults(maximos)
    transmition = consolidateResults(transmition)

    return {
        'medias': medias, 
        'minimos': minimos, 
        'maximos': maximos,
        'densidade': densidade,
        'densidade_anterior' : densidade_anterior,
        'transmition': transmition
    }

pastas = [d for d in os.listdir(dirPrefix) if d.startswith('results')]

print(pastas)

for pasta in pastas:
    result_centroide = processCentroideResults(pasta)
    result_random = processRandomResults(pasta)

    shutil.rmtree(pasta)
    os.mkdir(pasta)

    plot(pasta, result_centroide['medias'], result_random['medias'], "Tempo médio vs Número de nós", "average_time_vs_number_of_nodes.png", "centroide", "random")
    plot(pasta, result_centroide['minimos'], result_random['minimos'], "Tempo mínimo vs Número de nós", "minimum_time_vs_number_of_nodes.png", "centroide", "random")
    plot(pasta, result_centroide['maximos'], result_random['maximos'], "Tempo máxmo vs Número de nós", "maximum_time_vs_number_of_nodes.png", "centroide", "random")

    plot_density_comparison(pasta, result_centroide['densidade'], result_random['densidade'], "density_comparison_centroidexrandom.png", "Comparação de densidade Centroide x Random")

    plotWithoutLable(pasta, result_centroide['transmition'], result_random['transmition'], "Taxa de transmissão média vs Número de nós", "average_transmission_rate_vs_number_of_nodes.png", "centroide", "random")

#result_centroide = processCentroideResults()
#result_random = processRandomResults()

#plot(result_centroide['medias'], result_random['medias'], "Average Time vs Number of Nodes", "average_time_vs_number_of_nodes.png", "centroide", "random")
#plot(result_centroide['minimos'], result_random['minimos'], "Minimum Time vs Number of Nodes", "minimum_time_vs_number_of_nodes.png", "centroide", "random")
#plot(result_centroide['maximos'], result_random['maximos'], "Maximum TIme vs Number of Nodes", "maximum_time_vs_number_of_nodes.png", "centroide", "random")

#plot_density_comparison(result_centroide['densidade'], result_centroide['densidade_anterior'], "density_comparison_centroide.png")
#plot_density_comparison(result_random['densidade'], result_random['densidade_anterior'], "density_comparison_random.png")

#plot(result_centroide['transmition'], result_random['transmition'], "Average Transmission Rate vs Number of Nodes", "average_transmission_rate_vs_number_of_nodes.png", "centroide", "random")