# GraphSage
Inductive Representation Learning on Large Graphs

## 📄 Description
This project implements the GraphSAGE model with four types of aggregation (mean, max, sum, GCN) and allows comparison of its performance with DeepWalk across multiple datasets (Citeseer, PPI, and OpenAlex 🍀).
The project includes a comprehensive evaluation based on F1 score, recall, precision, accuracy, confusion matrix, and classification report, along with embedding visualization.

Implementation of GraphSAGE based on the paper "Inductive Representation Learning on Large Graphs" by Hamilton et al., 2017

## 📂 Project Structure

GraphSage/

├── Figures/               # All generated figures
├── graphvenv/             # Virtual environment to isolate dependencies
├── config.py              # Parameters and hyperparameters (dataset, learning rate, number of layers, etc.)
├── dataloader.py          # Data loading and preprocessing
├── models.py              # Model definitions: GraphSAGE (mean, max, LSTM aggregations), GCN, and DeepWalk
├── train.py               # Training loop with early stopping and logging
├── evaluation.py          # Evaluation and visualization functions (F1 score, recall, precision, confusion matrix, classification report)
├── utils.py               # Utility functions (embedding visualization, model saving, etc.)
├── requirements.txt       # Python dependencies (torch, torch-geometric, etc.)
├── .gitignore             # Files and folders to ignore in Git (e.g., __pycache__, checkpoints, logs, etc.)
├── README.md              # Project documentation (description, installation, usage, etc.)
└── main.py                # Entry point for training and evaluation

## 🛠️ Installation
### 📌 Prerequisites

Operating System: Windows (tested)
Hardware: A GPU is recommended for full training; compatible with platforms like SageMaker
Use a CUDA-compatible GPU for faster training
Python 3.12.6+

### Clone the repository

git clone https://github.com/SamarKri/GraphSage.git
cd GraphSage

### Create a virtual environment (Windows)

python -m venv graphvenv
graphvenv\Scripts\activate

### Install dependencies

pip install -r requirements.txt

### Run training with GraphSAGE (example on Citeseer)

python main.py --model graphsage --dataset citeseer

### 💡 Future Work

Add unit tests for dataloader, model, train, evaluation, and utils
Test implementation on other datasets like Cora, PubMed, or Reddit
Add a directory for dataset storage/reference (Citeseer, Cora, PubMed, Reddit, PPI, OpenAlex)
Optimize hyperparameters using Optuna
Define additional performance metrics such as AUC-ROC for multi-label problems
