```markdown
# NLU Assignment 2

## Prerequisites
* **Python 3.12+**
* **Install dependencies:** ```bash
  pip install -r problem_1/requirements.txt scikit-learn torch
  ```

## Problem 1: Word2Vec

```bash
cd problem_1

# 1. Add raw PDFs and URL .txt files to a 'content' directory first
mkdir content 

# 2. Extract text and generate corpus
python generate_corpus.py

# 3. Prepare dataset & generate word cloud
python prepare_dataset.py

# 4. Train SkipGram/CBOW and generate t-SNE visualizations
python word2vec_scratch.py
```

## Problem 2: Character-Level Name Generation

```bash
cd ../problem_2

# Note: Ensure the 'names.txt' dataset is present in this directory.

# 1. Train models and generate new names
python train_rnn.py
python train_blstm.py
python train_attention.py

# 2. Evaluate novelty and diversity
python evaluate.py
```
```