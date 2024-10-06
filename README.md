# LLM4TG

This project includes the source code for our paper "**Assessing Large Language Models for Cryptocurrency Transaction Graph Analysis: A Bitcoin Case Study**". 

## Contents

- [Dataset and Preprocess](#dataset-and-preprocess)
- [LLM4TG](#llm4tg)
- [CETraS](#cetras)
- [Experiment](#experiment)


## Dataset and Preprocess
The primary dataset used in the experiments is the **Bitcoin Address Subgraph Dataset** - [BASD-8](https://www.kaggle.com/datasets/lemonx/basd8) proposed in [our paper](https://ieeexplore.ieee.org/abstract/document/10020980). In addition, to explore the feature differences between graph-level and graph feature-level, we apply the same addresses to BASD-8 in the **Bitcoin Address Behavior Dataset** - [BABD-13](https://www.kaggle.com/datasets/lemonx/babd13) proposed in [our another paper](https://ieeexplore.ieee.org/abstract/document/10375557), implemented via `1-extract.py`. The corresponding LLM4TG format subgraph dataset is also available on [BASD-8](https://www.kaggle.com/datasets/lemonx/basd8).


## LLM4TG
(purpose and which file) The structure of the Bitcoin transaction graph inputting to LLMs is shown in ...


## CRTraS
(purpose and which file) This algorithm compressed the ... subgraphs 


## Experiment
(how to reproduce the results shown in Tables and Figures from levels 1 - 3) First, our experiments on BASD-8 cover transaction graph understanding levels 1 to 3, details can be found in `2-basd8-basic.py`, where `aux_querygpt` function is used for querying through OpenAI APIs.

