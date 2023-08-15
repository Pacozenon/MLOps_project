___
### ITESM Instituto Tecnológico de Estudios Superiores de Monterrey
### Course:     MLOps Machine Learning Operations
#### Teacher:   Carlos Mejia
#### Student:   Francisco Javier Torres Zenón  A01688757
____

## References
* Dataset and baseline notebook copied from [Online Payments Fraud Detection Dataset | Kaggle](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset) 
* Dataset: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset/download?datasetVersionNumber=1
* Baseline notebook: https://www.kaggle.com/code/nehahatti/online-payments-fraud-detection-project/notebook


# About this notebook
This notebook was taken and changed from [Kaggle](http://www.kaggle.com)

1. The task is to predict online payment fraud, given a number of features from online transfer/deposits transactions.

2. On Kaggle there were several notebooks related to this dataset(Decision Tree, Logistic Regresion, KNN, Gradient Boosting Classifier)
3. As a Baseline I choose one of the most accurated and simpler one, a notebook using the Decision Tree algorithm.

**Baseline Metrics**
```
Confussion Matrix
[[1270721     149]
 [     86    1568]]
```

```
 Classification Report
              precision    recall  f1-score   support

           0       1.00      1.00      1.00   1270870
           1       0.91      0.95      0.93      1654

    accuracy                           1.00   1272524
   macro avg       0.96      0.97      0.97   1272524
weighted avg       1.00      1.00      1.00   1272524
```


## Scope

* Project focused on MLOps, where the key concepts of ML frameworks learned on this course were applied in a holistic approach.

* In this project we will apply the best practices in MLOPs to a baseline notebook to create a model to predict online payment fraud, ready to use via API.

### Out of Scope

* Since we have already have on the baseline a good recall(0.95) and F1-Score(0.93) metrics over the FRAUD cases, we will note explore another methods.

* Also we will not make an intensive feature analysis nor feature engineering.


## Online Payments Fraud Detection
### Introduction
The introduction of online payment systems has helped a lot in the ease of payments. But, at the same time, it increased in payment frauds. Online payment frauds can happen with anyone using any payment system, especially while making payments using a credit card. 

That is why detecting online payment fraud is very important for credit card companies to ensure that the customers are not getting charged for the products and services they never paid. 



Part I 
    Definition
    Scope
    Baseline

Part II 
    Virtual environments
    Unit tests
    Pre-commits
    Refactoring
    Lining and formatting
    Directory structure
    OOP (Classes, methods, transformers, pipelines)
    REST API - FastAPI

This session talks about one of the most important practices to be able to climb an ML system: refactorization. Topics such as the directories structure of an ML system are included, the weaknesses that a notebook has to use in production, and a demo to refactorize an existing project.

## Setup
### Virtual environment

1. Create a virtual environment with `Python 3.10+` from the root folder
    * Create venv
        ```bash
        python3.10 -m venv venv310
        ```

    * Activate the virtual environment
        ```
        Linux:
        source venv310/bin/activate
        
        Windows:
            ./venv310/scripts/activate.ps1

        ```
2. Change to the refactored folder 
        ```
        Windows
        cd /Refactor/mlops_project/mlops_project

        ```
        
## Install all requerimients files

We have 3 requeriment files

* General packages for main program
        ```
        git install -r ../requirements-310.txt

        ```
* API packages
        ```
        git install -r ./requirements_api.txt

        ```
* PyTest packages
        ```
        git install -r ./requirements_dev.txt

        ```


### Cookiecutter
1. Ensure you cover all the [prerrequisites](https://cookiecutter.readthedocs.io/en/stable/installation.html#prerequisites)
2. Install cookiecutter
    ```bash
    python -m pip install cookiecutter
    ```
3. If you are working in the forked or cloned `itesm-mlops` project, please change the working directory to the session 9 with this command. 
    ```bash
    cd module-3/session-9
    ```

    Otherwise, if you are working on a new project, just skip this step and go to the next one.
3. Run this command, and fill out the information
    ```bash
    cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
    ```
    After completing the steps, a new project called [itesm_mlops](itesm_mlops) is going to be created.

## Directory structure
Once the template is created, it is time to create the folders with the code following this structure:
1. Define the different sections of the notebook
![Notebook](imgs/notebook.png)
2. Create the following folders with the scripts in it:
![Folders](imgs/folders.png)
    > **IMPORTANT!**  
    Create the `__init__.py` scripts on each folder where there are python scripts since they are important to mark the folder as a package, that will be imported later.
3. Migrate the notebook code to the scripts.

## Usage

1. Change the directory to `Refactor/mlops_project/mlops_project`.
2. Run `python mlops_project.py` in the terminal.

## Test API

1. Change the directory to `Refactor/mlops_project/mlops_project`.
2. Run `uvicorn api.main:app --reload` in the terminal.

##

## Checking endpoints
1. Access `http://127.0.0.1:8000/`, you will see a message like this `"Online Fraud Classifier is all ready to go!"`
2. Access `http://127.0.0.1:8000/docs`, the browser will display something like this:

    ![FastAPI Docs](./imgs/fast-api-docs.png)
    
3. Try running the sum endpoint by writing the values `3` and `5`, you will get the response body as follows
	
    **Response body**
    ```bash
    {
    "resultado": 8
    }
    ```
    ![sum](./imgs/sum.png)


## Activity
Follow the instructions in the [github.md](activity/github.md) file to complete the activity.
