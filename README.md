time2delivery
==============================

A short description of the project.

Project Organization
------------
.
├── Dockerfile -> File containing instructions to build a Docker Image with Flask API
├── LICENSE
├── Makefile
├── README.md
├── app.py -> Flask app with the model.
├── config
│   └── config_model.yaml -> config file with some variables.
├── data -> Folder with all data.
│   ├── predictions -> Folder with predictions made.
│   │   └── submission_predictions.csv -> File with predictions made by the. model.
│   ├── processed -> Folder with intermediate files 
│   │   ├── submission -> Submission file (rows in which total_minutes is NaN).
│   │   ├── test -> Test set.
│   │   ├── train -> Train Set.
│   │   └── train_best_features -> Train Set with the best features.
│   └── raw -> Folder with all files provided.
│       ├── all_orders.csv
│       ├── order_products.csv
│       ├── orders.csv
│       ├── shoppers.csv
│       └── storebranch.csv
├── models -> Folder with model and pipeline.
│   ├── feature_engineering_pipe.sav
│   └── model.sav
├── notebooks -> Folder with all notebooks developed.
│   ├── 1_data_acquisition.ipynb -> Data wrangling step.
│   ├── 2_data_cleaning.ipynb -> Data cleaning step.
│   ├── 3_eda_ft_engineering_ft_selection.ipynb -> EDA, Feature Engineering and Feature Selection steps.
│   └── 4_modeling.ipynb -> Modeling step.
├── request_time2delivery.py -> Example file using the Flask App.
├── requirements.txt -> Requirements for the project.
├── setup.py 
└── utils -> Module with all functions and classes created for the project.
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   ├── __pycache__
    │   └── cleaning.py
    ├── features
    │   ├── __init__.py
    │   ├── build.py
    │   ├── selection.py
    │   └── stats.py
    ├── models
    │   ├── __init__.py
    │   ├── evaluate.py
    │   ├── predict_model.py
    │   └── train_model.py
    └── visualization
        ├── __init__.py
        └── visualize.py
------------
