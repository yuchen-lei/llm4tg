import json
import os
from pprint import pprint

import pandas as pd
import sklearn.metrics


def process_results(results, llmkey):
    categories = set([v["ground_truth"] for v in results.values()])
    category2num = {v: k for k, v in enumerate(categories)}
    y_true = [category2num[v["ground_truth"]] for k, v in results.items()]
    y_pred = [category2num[v[llmkey]["result"][0]] for k, v in results.items()]
    acc = sum([v["ground_truth"] == v[llmkey]["result"][0] for k, v in results.items()])
    acctop3 = sum(
        [v["ground_truth"] in v[llmkey]["result"] for k, v in results.items()]
    )
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average=None)
    recall = sklearn.metrics.recall_score(y_true, y_pred, average=None)
    precision = sklearn.metrics.precision_score(y_true, y_pred, average=None)
    result_df = pd.DataFrame(
        {
            "label": category2num.keys(),
            "f1": f1,
            "recall": recall,
            "precision": precision,
        }
    )
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    precision = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
    marco_results = {
        "acc": acc / len(results),
        "acctop3": acctop3 / len(results),
        "f1": f1,
        "recall": recall,
        "precision": precision,
    }
    return marco_results, result_df


curdir = os.path.dirname(__file__)

print("ds_graph:")

with open(f"{curdir}/result_ds_graph.json") as f:
    results = json.load(f)
marco_results, result_df = process_results(results, "ds_graph")
pprint(marco_results)
pprint(result_df)


# pprint(marco_results)
# pprint(result_df)

# print("graph:")

# with open(f"{curdir}/result_gpt4o_graph.json") as f:
#     results = json.load(f)
# categories = set([v["ground_truth"] for v in results.values()])
# category2num = {v: k for k, v in enumerate(categories)}
# y_true = [category2num[v["ground_truth"]] for k, v in results.items()]
# y_pred = [category2num[v["gpt4o_graph"]["result"][0]] for k, v in results.items()]
# acc = sum(
#     [v["ground_truth"] == v["gpt4o_graph"]["result"][0] for k, v in results.items()]
# )
# acctop3 = sum(
#     [v["ground_truth"] in v["gpt4o_graph"]["result"] for k, v in results.items()]
# )
# f1 = sklearn.metrics.f1_score(y_true, y_pred, average=None)
# recall = sklearn.metrics.recall_score(y_true, y_pred, average=None)
# precision = sklearn.metrics.precision_score(
#     y_true, y_pred, average=None, zero_division=0
# )
# result_df = pd.DataFrame(
#     {
#         "label": category2num.keys(),
#         "f1": f1,
#         "recall": recall,
#         "precision": precision,
#     }
# )
# f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
# recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
# precision = sklearn.metrics.precision_score(
#     y_true, y_pred, average="macro", zero_division=0
# )

# marco_results = {
#     "acc": acc / len(results),
#     "acctop3": acctop3 / len(results),
#     "f1": f1,
#     "recall": recall,
#     "precision": precision,
# }
# pprint(marco_results)
# pprint(result_df)
