# Genetic Hyperparameters Search for Neural Networks

This project implements a **Genetic Algorithm-based hyperparameter search** to automatically find optimal neural network architectures (number of layers, units per layer, and activation functions) for a given classification task.  

It uses:
- **TensorFlow/Keras** for building and training models
- **Scikit-learn** for dataset generation and splitting
- **NumPy** for numerical operations
- **Genetic Algorithms** for evolutionary optimization
- **Meta-model prediction** to filter potential architectures before training (accelerates evolution)

---

## Features

- **Genetic Evolution** of neural network architectures
- **Merge and Split phases** for crossover
- **Mutation** for diversity in the population
- **Elitism** option for preserving top-performing architectures
- **Meta-model filtering** to avoid training unpromising candidates
- Works on **any classification dataset** (binary or multi-class)

---

## How It Works

1. **Initialization**  
   A population of random architectures is generated, each with:
   - Random number of layers
   - Random units per layer (within a specified range)
   - Random activation functions

2. **Evaluation**  
   Each architecture is trained on the dataset and its **accuracy** on the test set is used as its fitness score.

3. **Evolution Loop**  
   - **Merge Phase:** Combines architectures by averaging units in overlapping layers.
   - **Split Phase:** Takes partial layers from two parents and applies mutation.
   - **Meta-model Prediction:** Uses a small neural net to predict if a child is worth training.
   - **Natural Selection:** Keeps only the top-performing fraction of the population.

4. **Final Output**  
   The best-performing architecture is returned and retrained for final evaluation.

---

## Installation

```bash
git clone https://github.com/your-username/genetic-hyperparam-search.git
cd genetic-hyperparam-search
pip install -r requirements.txt
