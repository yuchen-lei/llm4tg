# LLM4TG

This project includes the source code for our work "**Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study**". If you find our work helpful for your research, please consider citing it as:

  @article{lei2025large,
    title={Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study},
    author={Lei, Yuchen and Xiang, Yuexin and Wang, Qin and Dowsley, Rafael and Yuen, Tsz Hon and Yu, Jiangshan},
    journal={arXiv preprint arXiv:2501.18158},
    year={2025}
  }

## Contents

- [Dataset and Preprocess](#dataset-and-preprocess)
- [LLM4TG](#llm4tg)
- [CETraS](#cetras)
- [Experiment](#experiment)
- [Acknowledgment](#acknowledgment)

## Dataset and Preprocess
The primary dataset used in the experiments is the **Bitcoin Address Subgraph Dataset** - [BASD-8](https://www.kaggle.com/datasets/lemonx/basd8) proposed in [our paper](https://ieeexplore.ieee.org/abstract/document/10020980). In addition, to explore the feature differences between graph-level and graph feature-level, we apply the same addresses to BASD-8 in the **Bitcoin Address Behavior Dataset** - [BABD-13](https://www.kaggle.com/datasets/lemonx/babd13) proposed in [our another paper](https://ieeexplore.ieee.org/abstract/document/10375557), implemented via `1-extract.py`. The corresponding LLM4TG format subgraph dataset is also available on [BASD-8](https://www.kaggle.com/datasets/lemonx/basd8).



## LLM4TG
(purpose and which file) The structure of the Bitcoin transaction graph inputting to LLMs is shown in ...


## CETraS
(purpose and which file) This algorithm compressed the ... subgraphs 


## Experiment
(how to reproduce the results shown in Tables and Figures from levels 1 - 3) First, our experiments on BASD-8 cover transaction graph understanding levels 1 to 3, details can be found in `2-basd8-basic.py`, where `aux_querygpt` function is used for querying through OpenAI APIs.


## Acknowledgment
This research is supported by OpenAI Researcher Access Program.


