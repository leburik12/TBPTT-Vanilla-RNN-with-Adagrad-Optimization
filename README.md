## Project Overview

This repository contains a minimal, from-scratch implementation of a Vanilla Recurrent Neural Network (RNN) for character-level language modeling. The setup is designed to investigate foundational sequence learning challenges, utilizing the Truncated Backpropagation Through Time (TBPTT) algorithm for efficient training and the Adagrad optimizer for adaptive parameter updates.
## 1. Scientific Implementation Details
### 1.1 Model Architecture (Vanilla RNN)

The model employs a single recurrent hidden layer with HIDDEN_SIZE=100 neurons. The core recurrence relation is:
ht​=tanh(Wxh​xt​+Whh​ht−1​+bh​)

    Input (xt​): One-hot encoded character vector (VOCAB_SIZE×1).

    Activation: The tanh non-linearity is used to stabilize hidden state magnitudes within [−1,1].

### 1.2 Training Dynamics (TBPTT & Regularization)

The training loop employs TBPTT for sequence management:

    Truncation Length (SEQ_LENGTH): T=25 steps. This dictates the maximum depth of the backpropagation chain, balancing computational cost with gradient preservation.

    Gradient Clipping: To mitigate the Exploding Gradient Problem, all calculated gradients (∇W,∇b) are clipped element-wise to the range [−5,5].

    Loss Function: Standard Cross-Entropy Loss is minimized across the batch sequence.

### 1.3 Optimization Algorithm (Adagrad)

The model is optimized using Adagrad (LEARNING_RATE=1×10−1), which maintains a separate, adaptive learning rate for every single parameter.
Wt+1​=Wt​−Gt​+ϵ​η​⋅gt​

This method is highly advantageous for the sparse nature of one-hot inputs but introduces a monotonic learning rate decay—a recognized limitation in long RNN training runs.
## 2. Getting Started and Reproducibility
### 2.1 Dependencies

    Python 3.x

    NumPy (Core library for all matrix operations)

### 2.2 Data Preparation

The model is trained on the file pride_and_prejudice.txt (or a similar large text corpus). The file must be present in the root directory.
### 2.3 Execution

The training and sampling loop is executed via the command line.

```bash
# Ensure pride_and_prejudice.txt is present in the current directory
$ python rnn_scratch.py 
```

## 3. Evaluation and Testing Protocol
### 3.1 Quantitative Evaluation (Perplexity)

The primary metric for validation is Perplexity (PPL), measured on a held-out validation set.

    Formula: PPL=eLavg​.

    Target Performance: A character-level PPL below 20 indicates strong generalization, suggesting the model is choosing among a small, plausible set of next characters.

### 3.2 Qualitative Evaluation (Contextual Inference)

Testing must involve controlled inference:

    Warm-up: Process a full sentence prompt (e.g., "It was a truth universally acknowledged...") using the forward pass to generate a final state hprompt​.

    Sampling: Begin stochastic sampling from the contextual state hprompt​ to test the model's ability to maintain narrative coherence.

## 4. Code Structure
This table details the functional role of each major component in the repository:

### ## 4. **Code Structure**

This table details the functional role of each major component in the repository:

| File/Function | Description | Scientific Purpose |
| :--- | :--- | :--- |
| `rnn_scratch.py` | Main script containing all training logic, parameter initialization, and the optimization loop. | Serves as the **minimal, executable definition** of the RNN computational graph and learning process. |
| `loss_fun()` | Implements the **Forward Pass**, calculates the Cross-Entropy Loss, and computes all necessary gradients via **Backpropagation Through Time (BPTT)**. Includes gradient clipping. | Defines the **objective function** and the method for calculating its derivative ($\nabla \mathcal{L}$), essential for optimization. |
| `sample()` | Implements the **stochastic inference** step where the next character index is chosen based on the output probability distribution ($\mathbf{P}(y_t \mid h_t)$). | Generates **qualitative results** (text samples) from the model's learned state, crucial for interpretability. |
| `rnn_weights_final.npz` | **(Generated)** Contains the final saved NumPy arrays for the model's parameters ($W, b$). | Provides a **permanent, version-controlled snapshot** of the trained model state for future deployment or analysis. |

## 5. Collaboration and Contributions

We welcome contributions focusing on architectural stability and performance (e.g., replacing tanh with ReLU or Adagrad with RMSprop).

    Branching: Create a dedicated feature branch from main.

    Pull Request: Submit a Pull Request detailing the scientific rationale, complexity trade-offs, and quantitative performance impact (PPL, stability) of your changes.
