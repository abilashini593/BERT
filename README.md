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
1. Clone the repository
git clone https://github.com/abilashini593/BERT.git
cd BERT/BERT-based_ATE_for_agriculture-main

2. Install dependencies
pip install -r requirements.txt

3. Prepare the dataset
python preprocessing/preprocess.py

4. Train the model
python train.py

5. Evaluate the model
python evaluate.py

🚀 Usage Instructions

Once your environment is ready:

Launch Jupyter Notebook:

jupyter notebook


Open the notebook:
BERT_ATE_Experiments.ipynb

Load the dataset and run preprocessing cells.

Select a fine-tuning configuration (learning rate, batch size, epochs).

Train the BERT model.

Evaluate performance using:

Precision

Recall

F1-Score

Compare multiple configurations to determine the best-performing setup.

Review the graphs and evaluation metrics generated in the notebook.
