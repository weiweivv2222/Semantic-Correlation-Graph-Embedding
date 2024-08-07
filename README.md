# Semantic Correlation Graph Embedding

This repository contains the code and resources related to the paper titled **"Semantic Correlation Graph Embedding"** by W.W. Wang, Y.C. Han, S. Bromuri, and M. Dumontier, presented at FUZZ-IEEE 2022.

## Overview

Many datasets include categorical features in the form of nominal and ordinal features. However, most machine learning algorithms cannot deal with categorical features directly because they require numerical input features. Categorical embeddings are an effective approach to converting categorical features into numerical vectors. This work proposes a novel embedding approach, called Semantic Correlation Graph Embedding, to create embeddings from knowledge graphs. The approach constructs a semantic correlation graph of triplets among the categorical features to learn numerical embeddings. Our approach aims to uncover relationships in categorical data in terms of low-level knowledge and semantics that may help group the features of the datasets under semantic entities. Three distinct embedding models are proposed according to how the graph is constructed. The results are evaluated with two public datasets and show that the learned embeddings produce a statistically significant improvement in the performance of the classification tasks in terms of AUC, F1 score, precision, and recall.

## Cite This Work

If you find our work useful in your research or applications, please consider citing our paper:

@inproceedings{645a48e1158742d48a4649bb5a0d0416,
title = "Semantic Correlation Graph Embedding",
author = "W.W. Wang and Y.C. Han and S. Bromuri and M. Dumontier",
year = "2022",
doi = "10.1109/FUZZ-IEEE55066.2022.9882620",
publisher = "IEEE",
booktitle = "2022 IEEE INTERNATIONAL CONFERENCE ON FUZZY SYSTEMS (FUZZ-IEEE)"
}

This research was supported by the Province of Limburg, The Netherlands, under grant number SAS-2020-03117. We would like to thank FUZZ-IEEE for providing a platform to present our work. Special thanks to our co-authors and collaborators for their valuable contributions.

## Getting Started

### Prerequisites

To run the code in this repository, you need to have the following software installed:
- Python 3.8
- Required Python packages (listed in `requirements.txt`)

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/weiweivv2222/Semantic-Correlation-Graph-Embedding.git

cd Semantic-Correlation-Graph-Embedding

pip install -r requirements.txt

### Create a virtual environment
```bash
conda create -m your_project_name
conda activate your_project_name



