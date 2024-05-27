import os
import json

curdir = os.path.dirname(__file__)

with open(f"{curdir}/results_fix2.json") as f:
    results = json.load(f)


def eq(a, b):
    if isinstance(a, str):
        a = float(a) if a else 0
    if isinstance(b, str):
        b = float(b) if b else 0
    return abs(a - b) < 1e-3


summary = {
    model: {
        key: [0, 0]  # [correct, total]
        for key in [
            "json_structure",
            "global_in_degree",
            "global_out_degree",
            "global_in_value",
            "global_out_value",
            "global_diff_degree",
            "global_diff_value",
            "node_in_degree",
            "node_out_degree",
            "node_in_value",
            "node_out_value",
            "node_special_info_a",
            "node_special_info_t",
        ]
    }
    for model in ["gpt4t", "gpt4o"]
}


def any2bool(val):
    val = str(val).lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def has_keys(d, keys):
    for key in keys:
        if key not in d:
            return False
    return True


def structural_santity_check(ans):
    if not has_keys(ans, ["global", "nodes"]):
        return False
    g = ans["global"]
    if not has_keys(
        g,
        [
            "max_in_degree_node",
            "max_out_degree_node",
            "max_in_value_node",
            "max_out_value_node",
            "max_diff_degree_node",
            "max_diff_value_node",
        ],
    ):
        return False
    n = ans["nodes"]
    if len(n) != 10:
        return False
    for node in n:
        if not has_keys(
            n[node],
            [
                "in_degree",
                "out_degree",
                "in_value",
                "out_value",
                "special_info",
            ],
        ):
            return False
    return True


for addr in results:
    ground_truth = results[addr]["ground_truth"]
    for model in ["gpt4t", "gpt4o"]:
        ans = results[addr][model]
        print(f"{addr=} {model=}")
        summary[model]["json_structure"][1] += 1
        if not structural_santity_check(ans):
            print(f"{addr} {model} failed structural santity check")
            continue
        summary[model]["json_structure"][0] += 1
        g = ans["global"]
        n = ans["nodes"]
        for key in ground_truth["global"]:
            if g[key] == ground_truth["global"][key]:
                summary[model][f"global_{key[4:-5]}"][0] += 1
            summary[model][f"global_{key[4:-5]}"][1] += 1
        for node in results[addr]["selected_nodes"]:
            if node not in n:
                print(f"{addr} {model} {node} not in nodes")
                continue
            for key in n[node]:
                if key == "special_info":
                    if isinstance(ground_truth["nodes"][node][key], list):
                        if any2bool(n[node][key]) == any2bool(
                            ground_truth["nodes"][node][key][1]
                        ):
                            summary[model][f"node_{key}_t"][0] += 1
                        summary[model][f"node_{key}_t"][1] += 1
                    else:
                        if eq(n[node][key], ground_truth["nodes"][node][key]):
                            summary[model][f"node_{key}_a"][0] += 1
                        summary[model][f"node_{key}_a"][1] += 1
                else:
                    if eq(n[node][key], ground_truth["nodes"][node][key]):
                        summary[model][f"node_{key}"][0] += 1
                    summary[model][f"node_{key}"][1] += 1

for model in ["gpt4t", "gpt4o"]:
    print(f"{model=}")
    for key in summary[model]:
        correct, total = summary[model][key]
        print(f"{key}: {correct}/{total} ({correct/total:.2%})")
    print()
import pandas as pd

df = pd.DataFrame(summary)
df = df.map(lambda x: f"{x[0]}/{x[1]} ({x[0]/x[1]:.2%})")
print(df)
