# Toxicity Classification

This is an academic project for detect/classify toxic text. The goal is to
compare various techniques for text classification and reach an acceptable
result for real-world usage.

## Dataset

The main target here is developing for classifying portuguese text. We have
found some open datasets appropriate for usage:

- OLID-BR
  - This is the main dataset we are using.
  - License: CC BY 4.0
  - URL: <https://www.kaggle.com/datasets/dougtrajano/olidbr>
  - Author: Douglas Trajano
  - Description: The Offensive Language Identification Dataset for Brazilian
    Portuguese (OLID-BR). It is composed of 7943 (extendable to 13,538)
    comments from different sources, such as YouTube and Twitter.

- ToLD-Br
  - License: CC BY-SA 4.0
  - URL: <https://github.com/JAugusto97/ToLD-Br>
  - Paper: <https://aclanthology.org/2020.aacl-main.91>
  - Authors: João A. Leite, Diego F. Silva, Kalina Bontcheva and Carolina Scarton
  - Description: The Toxic Language Detection Dataset for Brazilian Portuguese
    (ToLD-Br) is a dataset for the task of toxic language detection in Brazilian
    Portuguese composed of tweets.

## Dependencies

### Conda or Mamba

When using `conda`, `mamba` and its derivatives, you can create a new
environment, activate it, and add the dependencies with the following commands:

```bash
# Swap micromamba for conda if you are using it

# Create the environment
micromamba create -n toxicity-classification python=3.12 -c conda-forge

# Activate the environment
micromamba activate toxicity-classification

# Install dependencies for visualization
micromamba install jupyter polars numpy pandas pyarrow plotly matplotlib seaborn scipy -c conda-forge

# Install dependencies for text processing (stanza and nltk)
micromamba install stanza nltk -c conda-forge

# Install dependencies for machine learning (scikit-learn and pytorch)
micromamba install scikit-learn -c conda-forge

# For PyTorch, recommend checking for your specific setup at: https://pytorch.org
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For transformers, we use some pre-trained models from Hugging Face, so we need to install it
micromamba install transformers -c conda-forge
```

For installing everything at once, you can use the `environment.yml` file:

```bash
micromamba create -n toxicity-classification -f environment.yml
```
