## Overview

This repository contains code and resources for three different models developed for knowledge graph embedding. The models are:

1. **Model 01 - Cause Model**
2. **Model 02 - Cause and Prevent Model**
3. **Model 03 - Weighted Model**

The final results of KG2-2 are saved in `imp2/embedding_df_imp2.csv` for the second model, and those of KG1-2 are saved in `imp1/embedding_df_imp1.csv`.

### Directory Structure

- **my_task**: Contains files related to AWS TransE.
- **Thesis**: Contains raw data, all code, and intermediate results.

## Overall Process and Workflow

The development process involves the following steps:
1. Learn relations (e.g., using all job types to predict education levels).
2. Construct a knowledge graph.
3. Learn embeddings using AWS.

## Models Description

### Model 01 - Cause Model

- Considers only positive weights with a predefined threshold.

### Model 02 - Cause and Prevent Model

- Considers both positive and negative weights with a predefined threshold.

### Model 03 - Weighted Model

- Uses estimated coefficients to construct a weighted graph.

## Detailed Example - Model 03

### KG 2-1

1. Load bank data.
2. Drop numeric and date columns.
3. Run logistic regression models to get value pairs (e.g., all education types - all job types - weight).
4. Filter connections, keeping 8 types in this model.
5. Intermediate result saved as `result_df_sl_01.csv`.
6. Final triplet saved as `dgl_triplets_10.txt`.

### TransE

- Use the saved triplets as input for TransE.
- Output includes four files corresponding to entity and relation indices and embeddings.
- Example command to call TransE:

    ```bash
    DGLBACKEND=pytorch dglke_train --model_name TransE_l2 --data_path /Users/yuchen/my_task/data/ --dataset mywork --format raw_udd_hrt --data_files dgl_triplets5.txt --save_path /Users/yuchen/my_task --batch_size 1000 --neg_sample_size 200 --hidden_dim 10 --gamma 19.9 --lr 0.25 --max_step 500 --log_interval 100 --batch_size_eval 16 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 8
    ```

### KG 2-2

- Apply the knowledge graph embeddings to data.
- For each row with university and management, generate new features (10 features/columns).
- Input: TransE results.
- Output: CSV file with original data rows and corresponding embeddings (`imp2/embedding_df_imp2.csv`).

### KG 2-3

- Validate the impact of embeddings using a simple ML model.
- Input: Original data and embeddings.
- Output: Intermediate results (`measurements`, `measurements_em`).

## Function Explanations

- **preprocessing**: One-hot encoding for categorical features.
- **child_lr**: Logistic regression at the child level to predict categories based on other features.
- **father_lr**: Applies `child_lr` to all father categories, generating a list of value pairs.
- **afunction**: Concatenates value pairs into a string for triplets (e.g., "university causes management 1.3").

## Example Commands for TransE

### Models
Model 01

```bash
DGLBACKEND=pytorch dglke_train --model_name TransE_l2 --data_path /Users/yuchen/my_task/data/ --dataset mywork --format raw_udd_hrt --data_files dgl_triplets_01.txt --save_path /Users/yuchen/my_task --batch_size 1000 --neg_sample_size 200 --hidden_dim 10 --gamma 19.9 --lr 0.25 --max_step 500 --log_interval 100 --batch_size_eval 16 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 8
Model 02
DGLBACKEND=pytorch dglke_train --model_name TransE_l2 --data_path /Users/yuchen/my_task/data/ --dataset mywork --format raw_udd_hrt --data_files dgl_triplets_02.txt --save_path /Users/yuchen/my_task --batch_size 1000 --neg_sample_size 200 --hidden_dim 10 --gamma 19.9 --lr 0.01 --max_step 500 --log_interval 100 --batch_size_eval 16 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --has_edge_importance
Model 03
DGLBACKEND=pytorch dglke_train --model_name TransE_l2 --data_path /Users/yuchen/my_task/data/ --dataset mywork --format raw_udd_hrt --data_files adult_embedding_input_10.txt --save_path /Users/yuchen/my_task --batch_size 1000 --neg_sample_size 200 --hidden_dim 10 --gamma 19.9 --lr 0.01 --max_step 500 --log_interval 100 --batch_size_eval 16 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --has_edge_importance
