import sys

sys.path.append(".")

import os
from glob import glob

import networkx as nx

from llm4tg_repr import sample_single_graph


def sample_one_graph(graph_pth):
    graph = nx.read_graphml(graph_pth)
    sampled_graph = sample_single_graph(graph, 100)
    nx.write_graphml(sampled_graph, f"sampled_graphml/{os.path.basename(graph_pth)}")


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    subgraphs = glob("../llm4tg/basd-8/**/*.graphml", recursive=True)
    os.makedirs("sampled_graphml", exist_ok=True)
    list(
        tqdm(
            Parallel(n_jobs=-1, return_as="generator_unordered")(
                delayed(sample_one_graph)(graph_pth) for graph_pth in subgraphs
            ),
            total=len(subgraphs),
        )
    )
