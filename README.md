# Character-Level Name Generator

A character-level language model built entirely from scratch using NumPy. Given the previous few characters of a name, it predicts what character comes next — and by repeatedly sampling from its own predictions, it generates entirely new, plausible-sounding names it has never seen.

Trained on ~32,000 real first names.

## What's implemented

- **Vocabulary building**: mapping each of the 26 letters (plus a special `.` start/end marker) to an integer, and back
- **Training data construction**: sliding a fixed-size context window (3 characters) across every name to generate (context → next character) pairs — 228,146 examples total
- **Embedding table**: a learned lookup table mapping each character to a dense 10-dimensional vector, replacing one-hot encoding
- **Forward pass**: embedding lookup → flatten → Linear → ReLU → Linear → Softmax
- **Loss function**: cross-entropy
- **Backpropagation**: full manual gradient derivation, including gradient accumulation into the embedding table (`np.add.at`) since the same character can appear many times within a single batch
- **Training**: mini-batch gradient descent with per-epoch shuffling, plus learning rate decay once loss plateaued
- **Generation**: sampling from the model's output distribution (`np.random.choice`, not just `argmax`) to produce varied, novel names

## Architecture

```
context: 3 previous characters
   → Embedding lookup (3 x 10) → flatten (30)
   → Linear → ReLU
   → 100 (hidden layer)
   → Linear → Softmax
   → 27 (probability distribution over next character)
```

## Training details

| Hyperparameter      | Value        |
| ------------------- | ------------ |
| Context size        | 3 characters |
| Embedding dimension | 10           |
| Hidden layer size   | 100          |
| Batch size          | 64           |
| Learning rate       | 0.01         |
| Epochs              | 30           |

Loss dropped from ~3.18 (untrained baseline, ≈ `-log(1/27)`) down to ~2.08.

## Sample output

```
felyna
ady
nayir
teryana
dami
remylyn
```

## Setup

```bash
pip install numpy
jupyter notebook makemore.ipynb
```

The notebook downloads the training data directly from Karpathy's `makemore` repo on first run.
