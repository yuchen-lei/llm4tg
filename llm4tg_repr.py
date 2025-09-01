import math
import os
from io import StringIO

import networkx as nx
import numpy as np
from networkx import DiGraph


def summary_info_for_address(G: DiGraph, node):
    times = [x for _, _, x in G.to_undirected().edges(node, data="time")]
    return {
        "in_degree": G.in_degree(node),
        "out_degree": G.out_degree(node),
        "in_value": sum([x["value"] for _, _, x in G.in_edges(node, data=True)]),
        "out_value": sum([x["value"] for _, _, x in G.out_edges(node, data=True)]),
        "time_range": max(times) - min(times),
        # "out_nodes": set([t for _, t in G.edges(node)]),
    }


def summary_info_for_transaction(G: DiGraph, node):
    nodeview = G.nodes[node]
    return {
        "in_degree": nodeview["tx_inputs_count"],
        "out_degree": nodeview["tx_outputs_count"],
        "in_value": nodeview["tx_inputs_value"],
        "out_value": nodeview["tx_outputs_value"],
        "in_nodes": [f for f, _ in G.in_edges(node)],
        "out_nodes": [t for _, t in G.out_edges(node)],
    }


def sample_single_graph_repr_only(graph: nx.Graph):
    # in transaction graph, theres two types of nodes
    # 1. transaction nodes
    # 2. address nodes
    # transaction nodes have much more information
    # address nodes have only address
    # let's say we preserve most of the information of transaction nodes
    # but only caculate the in_ out_ info for address nodes
    sio = StringIO()
    dist = nx.single_source_shortest_path_length(
        graph.to_undirected(as_view=True), "n0"
    )
    dist_values = list(dist.values())
    dist_count = {v: dist_values.count(v) for v in set(dist_values)}
    last_dist = -1
    # to reduce tokens into 3000, we have to keep only about 75 nodes
    # we assume that nodes with larger value are more important
    # so we apply weighted sampling from nodes
    node_info = {
        node: (
            summary_info_for_transaction(graph, node)
            if (dist[node] & 1)
            else summary_info_for_address(graph, node)
        )
        for node in graph.nodes
    }
    # importance: log(in_value + out_value + 1) / (dist + 1)
    node_weights = [
        (
            math.log1p(node_info[node]["in_value"] + node_info[node]["out_value"])
            + 2
            * math.log1p(node_info[node]["in_degree"] + node_info[node]["out_degree"])
        )
        / (dist[node] + 1)
        for node in graph.nodes
    ]
    node_weights = 1 / np.array(node_weights)
    node_weights[0] = 0
    node_weights = node_weights / node_weights.sum()
    removed_nodes = np.random.choice(
        list(graph.nodes()),
        max(graph.number_of_nodes() - 60, 0),
        replace=False,
        p=node_weights,
    )
    removed_nodes = set(removed_nodes)
    node_keeps = set(graph.nodes()) - removed_nodes
    # for node in list(node_keeps):
    #     path = nx.shortest_path(graph.to_undirected(as_view=True), "n0", node)
    #     for node in path:
    #         node_keeps.add(node)
    pth2src = nx.single_source_shortest_path(graph.to_undirected(as_view=True), "n0")
    for knode in list(node_keeps):
        for pnode in pth2src[knode]:
            node_keeps.add(pnode)
    removed_nodes = set(graph.nodes()) - node_keeps
    graph.remove_nodes_from(removed_nodes)
    print(graph)
    for node in graph.nodes:
        node_type = "transaction" if (dist[node] & 1) else "address"
        if last_dist != dist[node]:
            print(
                f"Layer {dist[node]}: {dist_count[dist[node]]} {node_type} nodes",
                file=sio,
            )
            last_dist = dist[node]
        print(
            f"{node} {node_type}:",
            node_info[node],
            file=sio,
        )
    # method1(graph)
    graph_repr = sio.getvalue().replace("'", "")
    return graph_repr


def sample_single_graph(graph: nx.Graph, target_nodes: int):
    graph = graph.copy()
    # in transaction graph, theres two types of nodes
    # 1. transaction nodes
    # 2. address nodes
    # transaction nodes have much more information
    # address nodes have only address
    # let's say we preserve most of the information of transaction nodes
    # but only caculate the in_ out_ info for address nodes
    sio = StringIO()
    dist = nx.single_source_shortest_path_length(
        graph.to_undirected(as_view=True), "n0"
    )
    dist_values = list(dist.values())
    dist_count = {v: dist_values.count(v) for v in set(dist_values)}
    last_dist = -1
    # to reduce tokens into 3000, we have to keep only about 75 nodes
    # we assume that nodes with larger value are more important
    # so we apply weighted sampling from nodes
    node_info = {
        node: (
            summary_info_for_transaction(graph, node)
            if (dist[node] & 1)
            else summary_info_for_address(graph, node)
        )
        for node in graph.nodes
    }
    # importance: log(in_value + out_value + 1) / (dist + 1)
    node_weights = [
        (
            math.log1p(node_info[node]["in_value"] + node_info[node]["out_value"])
            + 2
            * math.log1p(node_info[node]["in_degree"] + node_info[node]["out_degree"])
        )
        / (dist[node] + 1)
        for node in graph.nodes
    ]
    node_weights = 1 / np.array(node_weights)
    node_weights[0] = 0
    node_weights = node_weights / node_weights.sum()
    removed_nodes = np.random.choice(
        list(graph.nodes()),
        max(graph.number_of_nodes() - target_nodes, 0),
        replace=False,
        p=node_weights,
    )
    removed_nodes = set(removed_nodes)
    node_keeps = set(graph.nodes()) - removed_nodes
    # for node in list(node_keeps):
    #     path = nx.shortest_path(graph.to_undirected(as_view=True), "n0", node)
    #     for node in path:
    #         node_keeps.add(node)
    pth2src = nx.single_source_shortest_path(graph.to_undirected(as_view=True), "n0")
    for knode in list(node_keeps):
        for pnode in pth2src[knode]:
            node_keeps.add(pnode)
    removed_nodes = set(graph.nodes()) - node_keeps
    graph.remove_nodes_from(removed_nodes)
    for node in graph.nodes:
        node_type = "transaction" if (dist[node] & 1) else "address"
        if last_dist != dist[node]:
            print(
                f"Layer {dist[node]}: {dist_count[dist[node]]} {node_type} nodes",
                file=sio,
            )
            last_dist = dist[node]
        print(
            f"{node} {node_type}:",
            node_info[node],
            file=sio,
        )
    # method1(graph)
    graph_repr = sio.getvalue().replace("'", "")
    return graph_repr, graph


def graph_full_repr(graph: nx.DiGraph):
    # in transaction graph, theres two types of nodes
    # 1. transaction nodes
    # 2. address nodes
    # transaction nodes have much more information
    # address nodes have only address
    # let's say we preserve most of the information of transaction nodes
    # but only caculate the in_ out_ info for address nodes
    # graph: DiGraph = networkx.read_graphml(subgraph_file)
    sio = StringIO()
    dist = nx.single_source_shortest_path_length(graph.to_undirected(), "n0")
    dist_values = list(dist.values())
    dist_count = {v: dist_values.count(v) for v in set(dist_values)}
    last_dist = -1
    # to reduce tokens into 3000, we have to keep only about 75 nodes
    # we assume that nodes with larger value are more important
    # so we apply weighted sampling from nodes
    node_info = {
        node: (
            summary_info_for_transaction(graph, node)
            if (dist[node] & 1)
            else summary_info_for_address(graph, node)
        )
        for node in graph.nodes
    }
    # importance: log(in_value + out_value + 1) / (dist + 1)
    for node in graph.nodes:
        node_type = "transaction" if (dist[node] & 1) else "address"
        if last_dist != dist[node]:
            print(
                f"Layer {dist[node]}: {dist_count[dist[node]]} {node_type} nodes",
                file=sio,
            )
            last_dist = dist[node]
        print(
            f"{node} {node_type}:",
            node_info[node],
            file=sio,
        )
    # method1(graph)
    graph_repr = sio.getvalue().replace("'", "")
    return graph_repr
