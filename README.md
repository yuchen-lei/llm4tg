# LLM4TG

This project includes the source code for our work **[Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study](https://arxiv.org/abs/2501.18158)**. If you find our work helpful for your research, please consider citing it as:

    @article{lei2025large,
      title={Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study},
      author={Lei, Yuchen and Xiang, Yuexin and Wang, Qin and Dowsley, Rafael and Yuen, Tsz Hon and Yu, Jiangshan},
      journal={arXiv preprint arXiv:2501.18158},
      year={2025}
    }

If you have any questions please feel free to contact by e-mail at Yuexin.Xiang@monash.edu.

## Contents

- [Dataset](#dataset)
- [Layered Assessment Framework](#layered-assessment-framework)
- [LLM4TG Format](#llm4tg-format)
- [CETraS Algorithm](#cetras-algorithm)
- [Additional Notes](#additional-notes)
- [Acknowledgment](#acknowledgment)

## Dataset 
### 1. Bitcoin Address Subgraph Dataset ([BASD-8](https://www.kaggle.com/datasets/lemonx/basd8))  
BASD-8 serves as our primary dataset for experiments. It captures *transactional subgraph structures* of Bitcoin addresses and was introduced in our [**IEEE Big Data'22** paper](https://ieeexplore.ieee.org/abstract/document/10020980).

### 2. Bitcoin Address Behavior Dataset ([BABD-13](https://www.kaggle.com/datasets/lemonx/babd13))  
BABD-13 is used to explore the differences between *raw graph structures* and *graph-derived features*. The subset of this dataset used in this work (i.e., `babd13-slim.csv`) contains the same addresses as BASD-8 but focuses on *behavioral patterns* derived from Bitcoin transactions. It was introduced in our [**IEEE TIFS'24** paper](https://ieeexplore.ieee.org/abstract/document/10375557).


## Layered Assessment Framework

We propose a **three-level framework** for measuring the understanding of a transaction graph. The following illustration and descriptions provide an overview of the proposed framework and its levels:

---

### **Level 1 - Foundational Metrics**  
Using LLMs to determine the **basic information** of the graph (see `lv1_nodebasic`), such as:  
- *In-degree* of a node  
- *Output token amount* of a node  

---

### **Level 2 - Characteristic Overview**  
Applying LLMs to identify **key characteristics** of the graph (see `lv2_characteristic`), for example:  
- A node with a *significantly large out-degree*  
- A node that transfers a *significantly large total amount of tokens*

---

### **Level 3 - Contextual Interpretation**  
Utilizing LLMs to classify *cryptocurrency address types* for *unlabeled addresses* by leveraging labeled address samples (see `lv3_categorize`).  

---


## LLM4TG Format

The *format conversion* is handled by the function **`graph_full_repr`** within the file `llm4tg_repr.py`. 

Additionally, the **dataset in the LLM4TG format** for subgraphs is also available in the [BASD-8](https://www.kaggle.com/datasets/lemonx/basd8).


## CETraS Algorithm
Despite LLM4TG's efficiency, some transaction graphs are too large for tasks like classification that involve few-shot learning, which processes multiple graphs at once. To tackle this, we introduce CETraS, a method that condenses mid-sized transaction graphs while maintaining essential structures. The algorithm (see **`sample_single_graph_repr_only`** within the file `llm4tg_repr.py`).


## Additional Notes

`aux_querygpt.py` is used to interact with OpenAI APIs for querying purposes.

`lv2-quality-note.pdf` contains the empirical analysis results for the characteristic overview.

`ver1` is the previous version of the implementation.

## Acknowledgment
This research is supported by [**OpenAI Researcher Access Program**](https://openai.com/form/researcher-access-program) (Project ID: 0000007730).


