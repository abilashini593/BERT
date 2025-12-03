# BERT-Based Agricultural Term Extraction (ATE)

A research project investigating how different fine-tuning configurations impact the performance of BERT for extracting agricultural domain-specific terms.

## Features

Evaluates multiple fine-tuning configurations of BERT

Preprocessing pipeline for agricultural text data

Structured Jupyter Notebooks for experimentation

Performance comparison using Precision, Recall, F1-Score

Modular code for easy reproducibility

Visualizations for comparing model metrics

## Tech Stack

Model: BERT (Bidirectional Encoder Representations from Transformers)

Framework: PyTorch, HuggingFace Transformers

Environment: Jupyter Notebook

Language: Python

Tools: NumPy, Pandas, Matplotlib, Scikit-learn

## Getting Started

Clone the repository

git clone https://github.com/abilashini593/BERT.git
cd BERT/BERT-based_ATE_for_agriculture-main


Install dependencies

pip install -r requirements.txt


Prepare the dataset

python preprocessing/preprocess.py


Train the model

python train.py


Evaluate the model

python evaluate.py


## Usage Instructions

Follow the steps below to run experiments and evaluate different fine-tuning configurations:

1.Start Jupyter Notebook

2.Open the Project Notebook
Launch BERT_ATE_Experiments.ipynb from the Jupyter interface.

3.Load and Preprocess the Dataset
Run all cells in the data loading and preprocessing sections.

4.Choose a Fine-Tuning Configuration
Select hyperparameters such as:

-Learning rate

-Batch size

-Number of epochs

5.Train the BERT Model
Execute the training cells to fine-tune the model on the agricultural term extraction dataset.

6.Evaluate Model Performance
Review the computed metrics:

Precision

Recall

F1-Score

7.Compare Configurations
Run experiments using different hyperparameter combinations and analyze the results to identify the best-performing setup.

8.Review Outputs
Examine generated graphs, logs, and evaluation tables to understand model behavior and experiment outcomes.
