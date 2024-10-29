# Overview
This package aims to help analysising biomedical data using ML method in python.    

![image](./documents/images/workflow/PineBioML_workflow_v3.png) 

# System requirements
   1. Python 3.10+
   2. The following python module dependencies are required: 
   > pandas openpyxl xlrd tqdm seaborn gprofiler-official jupyter jupyterlab optuna scikit-learn umap-learn pacmap statsmodels mljar-supervised joblib


# Installation

### 1. Install Python
Please follow the tutorial to install python (the sections "Visual Studio Code" and "Git" are optional):

    https://learn.microsoft.com/en-us/windows/python/beginners 
    
Please skip this step if you already have python 3.10+ installed in your PC.

### 2. Install dependencies and execute the scripts

Step 1. Download the examples from Release and unzip it.

    https://github.com/ICMOL/undetermined/releases

Step 2. Please open Windows PowerShell, and execute the following command.

    pip install PineBioML          

Step 3. Move to the directory of the unzipped examples, and open the jupyter interface. 

Please execute the following command to open jupyter. You will see the figure, if the scripts execute correctly.

    > python -m notebook


![image](./documents/images/tutorial/browser_jupyter.png)


# Input Table Format

The input data should be tabular and placed in the ./input folder. We accept .csv, .tsv, .xlsx and R-table in .txt formats.    

# Process

### 0. Document

[API](https://htmlpreview.github.io/?https://github.com/ICMOL/PineBioML/blob/main/documents/API/index.html)

### 1. Missing value preprocess
|        ID         |        Option         |  Definition |
|---------------------|----------------|------------------------------|
|  1 | Deletion              | Remove the features that are too empty.     |
|  2 | Imputation with a constant value  | Impute missing values with a constant value, such as 0 or the feature mean. |
|  3 | Imputation using K-NN algorithm        | Impute missing values with the mean or median of the k nearest samples. |


### 2. Data transformation
|        ID         |        Option         |  Definition |
|---------------------|----------------|------------------------------|
|  1 | PCA              | Principal component transform.    |  |
|  2 | Power transform  | To make data more Gaussian-like, you can use either Box-Cox transform or Yeo-Johnson transform. |   |
|  3 | Feature clustering        | Group similar features into a cluster.  |  |
|  4 | Feature expansion        | Generating new features by add/product/ratio in random pair of existing features.  |  |


### 3. Feature selection
|        ID         |        Option         |  Definition |
|---------------------|----------------|------------------------------|
|  1 | Volcano plot     | Selecting by group p-value and fold change   |  |
|  2 | Lasso regression | Selecting by Linear models with L1 penalty |   |
|  3 | Decision stump   | Selecting by 1-layer decision tree  |  |
|  4 | Random Forest    | Selecting by Gini impurity or permutation importance over a Random Forest |  |
|  5 | AdaBoost         | Selecting by Gini impurity over a AdaBoost model  |  |
|  6 | Gradient boosting| Selecting by Gini impurity over a gradient boosting, such as XGboost or LightGBM  |  |
|  7 | Linear SVM              | Selecting by support vector from support vector machine |  |


### 4. Model building
|        ID         |        Option         |  Definition |
|---------------------|----------------|------------------------------|
|  1 | ElasticNet    | Using Optuna to find a not-bad hyper parameters on given dataset.   |  |
|  2 | SVM       | Using Optuna to find a not-bad hyper parameters on given dataset. |   |
|  3 | Decision Tree | Using Optuna to find a not-bad hyper parameters on given dataset.  |  |
|  4 | Random Forest | Using Optuna to find a not-bad hyper parameters on given dataset.  |  |
|  5 | AdaBoost | Using Optuna to find a not-bad hyper parameters on given dataset.  |  |
|  6 | XGBoost | Using Optuna to find a not-bad hyper parameters on given dataset.  |  |
|  7 | LightGBM | Using Optuna to find a not-bad hyper parameters on given dataset.  |  |
|  8 | CatBoost | Using Optuna to find a not-bad hyper parameters on given dataset.  |  |

### 5. Report and visualization
|        ID         |        Option         |  Definition |
|---------------------|----------------|------------------------------|
|  1 | data_overview  | Giving a glance to input data.   |  |
|  2 | classification_summary | Summarizing a classification task |  |

# Examples for Program Demonstration    

Chosse one of the following examples, double click it in jupyter interface:    
| ID |     Name      |       Description                |
|----|---------------|----------------------------------|
|  1 | example_BasicUsage.ipynb   | Demonstrate the basic features of PineBioML  |  |
|  2 | example_Proteomics.ipynb         | An example on proteomics data analysis |  |
|  3 | example_PipeLine.ipynb           | Demonstrate how to use the pipeline to store the whole data processing flow |  |
|  4 | example_Pine.ipynb               | Demonstrate how to use Pine ml to finding the best data processing flow in an efficient way |  |
|  5 | example_UsingExistingModel.ipynb | An example of unsing existing models/pipeline gained from 3. , 4. or 5.  |  |



Click the buttom and the script should start.
![image](./documents/images/tutorial/jupyter_runall.png)


# Cites
The example data is from [LinkedOmicsKB](https://kb.linkedomics.org/)
>  **A proteogenomics data-driven knowledge base of human cancer**, Yuxing Liao, Sara R. Savage, Yongchao Dou, Zhiao Shi, Xinpei Yi, Wen Jiang, Jonathan T. Lei, Bing Zhang, Cell Systems, 2023.
