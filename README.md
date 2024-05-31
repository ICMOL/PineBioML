# Overview
(麻煩用中文寫三百字引言,介紹軟體目的)
This package aims to help analysis biomedical data using ML method in python.

![image](./documents/images/workflow/auto_selection_workflow.png) 

# System requirements
   1. Python 3.9+
   2. The following dependencies are required: numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm, jupyter, lightgbm, xgboost


# Installation

### 1. Install Python
Please follow the tutorial to install python (the "Visual Studio Code" and "Git" are optional):

    https://learn.microsoft.com/en-us/windows/python/beginners 
    
Please skip this step, if you have python 3.9+ installed in your PC.

### 2. Install dependencies and execute the scripts

Step 1. Download our scripts from Release and unzip it.

    https://github.com/ICMOL/undetermined/releases

Step 2. Install dependencies: Please open Windows PowerShell, move to the directory of our scripts, and execute the following command.

    > pip install -r ./requirements.txt          

Step 3. Open the jupyter interface

Please execute the following command to open jupyter. You will see the figure, if the scripts execute correctly.

    > jupyter notebook    

![image](./documents/images/tutorial/browser_jupyter.png)


# Input Table Format

The input data should be tabular and placed in the ./input folder. We accept .csv, .tsv, .xlsx and R-table in .txt formats.

# Process

[API](./documents/API/index.html)

### 1. Missing value imputation
|        ID         |        Option         |  Definition |
|---------------------|----------------|------------------------------|
|  1 | Deletion              | Delete the features that contain too many missing values |
|  2 | Imputation with a constant value  | Impute missing value with a constant value. For example, 0 or feature mean |
|  3 | Imputation using K-NN algorithm        | Impute missing value with the mean/median of k nearest samples |
|  4 | Self regression        |  Training a self regression model to predict the missing value  |


### 2. Data transformation
|        ID         |        Option         |  Definition |
|---------------------|----------------|------------------------------|
|  1 | PCA              | Principal component transform    |  |
|  2 | Power transofmation  | Box-Cox transform or yeo-johnson transform to make data more gaussian-like |   |
|  3 | Feature bagging        | Put similar features into a bag  |  |


### 3. Feature Selection
|        ID         |        Option         |  Definition |
|---------------------|----------------|------------------------------|
|  1 | Volcano plot  | Seleting by group p-value or fold change   |  |
|  2 | Lasso regression | Seleting by L1 penalty |   |
|  3 | Decision stump        | Seleting by 1-layer decision tree  |  |
|  4 | Random Forest        | Seleting by Gini impuracy or permutation importance over a Random Forest |  |
|  5 | AdaBoost        | Seleting by Gini impuracy over a AdaBoost model  |  |
|  6 | Gradient boosting        | Seleting by Gini impuracy over a Gradient boosting such as XGboost or LightGBM  |  |
|  7 | SVM        | Selectiing by support vector of a trained support vector machine |  |


### 4. Model building
 - hyper parameter tuner (by optuna)
 - 

### 5. Report and visualization


# An Example for Program Demonstration    

<請描述你提供的資料內容是甚麼>

Chosse one of the following examples, double click it in jupyter interface:    
- example_feature_selection.ipynb

![image](./documents/images/tutorial/jupyter_runall.png)
click the buttom and the script should start.

