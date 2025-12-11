TBPTT-Vanilla-RNN-with-Adagrad-Optimization: Character-Level Language Modeling
Project Overview

This repository contains a minimal, from-scratch implementation of a Vanilla Recurrent Neural Network (RNN) for character-level language modeling. The setup is designed to investigate foundational sequence learning challenges, utilizing the Truncated Backpropagation Through Time (TBPTT) algorithm for efficient training and the Adagrad optimizer for adaptive parameter updates.
Implementation Details

Model Architecture (Vanilla RNN)

The model employs a single recurrent hidden layer with HIDDEN_SIZE=100 neurons. The core recurrence relation is:
ht​=tanh(Wxh​xt​+Whh​ht−1​+bh​)
    Input (xt​): One-hot encoded character vector (VOCAB_SIZE×1).
    Activation: The tanh non-linearity is used to stabilize hidden state magnitudes within [−1,1].

 Training Dynamics (TBPTT & Regularization)

The training loop employs TBPTT for sequence management:
    Truncation Length (SEQ_LENGTH): T=25 steps. This dictates the maximum depth of the backpropagation chain, balancing computational cost with gradient preservation.
    Gradient Clipping: To mitigate the Exploding Gradient Problem, all calculated gradients (∇W,∇b) are clipped element-wise to the range [−5,5].
    Loss Function: Standard Cross-Entropy Loss is minimized across the batch sequence.

 Optimization Algorithm (Adagrad)

The model is optimized using Adagrad (LEARNING_RATE=1×10−1), which maintains a separate, adaptive learning rate for every single parameter.
Wt+1​=Wt​−Gt​+ϵ​η​⋅gt​

This method is advantageous for the sparse nature of one-hot inputs but introduces a monotonic learning rate decay, a recognized limitation in long RNN training runs.
