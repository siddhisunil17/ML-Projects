# 🧠 Handwritten Digit Classification (ML Project 1)

## 📌 Project Overview

This project implements a neural network from scratch using Python to classify handwritten digits from the MNIST dataset. The goal is to build a digit classifier without using any deep learning libraries like TensorFlow or PyTorch.

The neural network consists of an input layer, one hidden layer, and an output layer. We use forward propagation, backpropagation, and gradient descent to optimize weights.

---

## 📁 Project Structure

- `nnScript.py` : Core implementation of neural network training and testing pipeline
- `params.pickle` : Serialized weights and biases from trained model
- `Project_1_description_updated.pdf` : Assignment problem statement and implementation details
- `mnist_all.mat` : Input data file (not included due to large size – provide locally)
- `README.md` : This file
- `requirements.txt` : Python dependencies

---

## 🚀 Getting Started

### 1️⃣ Install Python Dependencies

pip install -r requirements.txt
---

## 🧠 Features

- Neural Network with 1 hidden layer

- Implements backpropagation and gradient descent from scratch

- Support for command-line arguments to control training and testing

- Evaluation with confusion matrix and classification accuracy
---
## 📂 Notes
- Code avoids using any machine learning frameworks

- Data is loaded from MATLAB .mat file format

- The params.pickle file stores the learned weights and biases for reuse
