# %%
import json
import pickle
import numpy as np
import pandas as pd

results = pickle.load(open("basd-8-results.pkl", "rb"))


# calucate the hallucination, we define hallucination as
# in_degree or out_degree is zero in ground truth, but non-zero in llm's output

# model = "gpt-4-1106-preview"
model = "gpt-3.5-turbo-1106"


def have_hallucination(ground_truth, llm_result):
    return any(
        [
            bool(ground_truth["in_degree"]) ^ bool(llm_result["in_degree"]),
            bool(ground_truth["out_degree"]) ^ bool(llm_result["out_degree"]),
        ]
    )


hallucination_count = sum(
    [
        have_hallucination(x["ground_truth"], y)
        for x in results
        for y in x["llm_result"][model]
    ]
)
print(hallucination_count)
# we find the answer is 0 (unexpected)
from pprint import pprint

# calc the distribution for the difference of each feature
feature_list = ["in_degree", "out_degree", "in_value", "out_value", "first_time"]
result_df = []
# defaults = {k: np.mean([x["ground_truth"][k] for x in results]) for k in feature_list}
for feature in feature_list:
    diff_abs = [
        x["ground_truth"][feature] - y[feature]
        for x in results
        for y in x["llm_result"][model]
        if feature in y
    ]
    diff_rel = [
        (x["ground_truth"][feature] - y[feature]) / max(x["ground_truth"][feature], 1)
        for x in results
        for y in x["llm_result"][model]
        if feature in y
    ]
    result_map = {
        "feature": feature,
        "diff_abs_mean": np.mean(np.abs(diff_abs)),
        "diff_abs_std": np.std(diff_abs),
        "diff_rel_mean": np.mean(np.abs(diff_rel)),
        "diff_rel_std": np.std(diff_rel),
    }
    print(feature)
    print(sum(np.abs(diff_abs) < 1e-2) / len(diff_rel) * 100)
    print(sum(np.abs(diff_rel) < 0.05) / len(diff_rel) * 100)
    result_df.append(result_map)
    # pprint(result_map)
df = pd.DataFrame(result_df)
print(df)


# %%
for feature in feature_list:
    max_truth = max([x["ground_truth"][feature] for x in results])
    min_truth = min([x["ground_truth"][feature] for x in results])
    max_llm = max(
        [y[feature] for x in results for y in x["llm_result"]["gpt-4-1106-preview"]]
    )
    min_llm = min(
        [y[feature] for x in results for y in x["llm_result"]["gpt-4-1106-preview"]]
    )
    print(feature)
    print(max_truth, min_truth, max_llm, min_llm)

# %%
