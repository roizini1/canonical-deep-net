unsupervised-learning
==============================

Sparse Canonical Correlation Analysis

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- OPTIONAL: Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <-   OPTIONAL: Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── conf
    │   │   │   ├── __init__.py
    │   │   │   ├── model_config.yaml
    │   │   │   └── config.py
    │   │   ├── __init__.py
    │   │   ├── train_model.py
    │   │   ├── predict_model.py
    │   │   ├── lightning_net.py
    │   │   ├── stg.py
    │   │   ├── subnets.py
    │   │   ├── sdcca.py
    │   │   └── outputs     <- yaml file written by hydra-core - an configuration file of train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   ├── __init__.py
    │   │   ├── Gates visualization.png
    │   │   ├── left_gate.pt
    │   │   ├── right_gate.pt
    │   │   ├── visual gates.py
    │   │   └── visualize.py
    │
    ├── spec-file.txt   <- The requirements file for reproducing the environment, e.g.
    │                         generated with `conda list --explicit > spec-file.txt`
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
