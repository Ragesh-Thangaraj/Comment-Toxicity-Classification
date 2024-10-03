# Comment-Toxicity-Classification

# Toxic Comment Classification Using Bidirectional LSTM and GloVe Embeddings

## Overview

This project aims to classify toxic comments using a Bidirectional LSTM network powered by GloVe word embeddings. It is based on the dataset from the Jigsaw Toxic Comment Classification Challenge, where comments are labeled with multiple forms of toxicity, including `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. The goal is to develop a deep learning model that can detect toxicity and output the appropriate labels for a given comment.

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Description](#model-description)
4. [Training](#training)
5. [Testing and Evaluation](#testing-and-evaluation)
6. [Gradio Interface](#gradio-interface)
7. [How to Use](#how-to-use)
8. [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.x
- TensorFlow
- Pandas
- Numpy
- Gradio

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt

The requirements.txt file should include:
tensorflow
pandas
numpy
gradio
matplotlib
scikit-learn

GloVe Embeddings
Download the GloVe embeddings (Link: https://nlp.stanford.edu/projects/glove/). Ensure you use the glove.6B.100d.txt or glove.6B.200d.txt file based on the model configuration.

Dataset
The dataset is from the Jigsaw Toxic Comment Classification Challenge in Kaggle.(Link: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
Download the dataset and place it in the jigsaw-toxic-comment-classification-challenge/ folder.

train.csv contains the training data with columns id, comment_text, toxic, severe_toxic, obscene, threat, insult, and identity_hate.

Model Description

Model Architecture
The model uses the following architecture:

Embedding Layer: Pre-trained GloVe embeddings (100/200-dimensional).
Bidirectional LSTM: 64 units for capturing dependencies in both forward and backward directions.
Dense Layers: Fully connected layers with ReLU activation.
Output Layer: Sigmoid activation for each label to handle multi-label classification.

Training
The model is trained using:

Binary Cross-Entropy Loss: Due to the multi-label classification.
Adam Optimizer.
Early Stopping: To prevent overfitting.

Evaluation
Metrics: Accuracy and loss.
Test Set: Performance on unseen data is evaluated using accuracy.

How to Use
Enter a comment in the Gradio interface.
Get predictions: The model will output a dictionary of labels (toxic, severe_toxic, obscene, threat, insult, and identity_hate) with True/False values indicating whether each label is triggered.

Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue.
### **Conclusion:**
This README provides an overview of the project, installation steps, instructions for using the model, and information about the Gradio interface. It guides users through the steps to set up the project, run the model, and test the predictions interactively.
