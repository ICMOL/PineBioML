# Overview
(麻煩用中文寫三百字引言,介紹軟體目的)
This package aims to help analysis biomedical data using ML method in python.


![image](./images/workflow/auto_selection_workflow.png) 

# System requirements
   1. Python 3.9+
   2. The following dependencies are required: numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm, jupyter, lightgbm, xgboost


# Installation

### 1. Install Python
Please follow the tutorial to install python (the "Visual Studio Code" and "Git" are optional):

    https://learn.microsoft.com/en-us/windows/python/beginners 
    
Please skip this step, if you have python 3.9+ installed in your PC.

### 2. Install dependencies and execute the scripts
Step 1. Download our scripts from Release

    https://github.com/ICMOL/undetermined/releases

Step 2. Install dependencies: Please open Windows PowerShell, move to the directory of our scripts, and execute the following command.

    > pip install -r ./requirements.txt          

Step 3. Open the interface of python (ipython)

Please execute the following command to open ipython. You will see the figure, if the scripts execute correctly.

    > jupyter notebook    

![image](./images/tutorial/browser_jupyter.png)


# Input Table Format

The input data should be tabular and placed in the ./input folder. We accept .csv, .tsv, and .xlsx formats as input.

*Note: A Sample ID is required. You can designate a column name as the indexing ID. An error will occur if the IDs of x and y do not match. (甚麼是x,甚麼是y?)


# Process

### 1. Missing value imputation
|        ID         |        Option         |  Definiation |
|---------------------|----------------|------------------------------|
|  1 | Deletion              | (請用英文說明)    |
|  2 | Imputation with a constant value  | (請用英文說明) |
|  3 | Imputation using K-NN algorithm        | (請用英文說明)  |
|  4 | Self regression        |  (請用英文說明)  |


### 2. Data transformation
|        ID         |        Option         |  Definiation |
|---------------------|----------------|------------------------------|
|  1 | PCA              | (請用英文說明)    |  |
|  2 | Power transofmation  | (請用英文說明) |   |
|  3 | Feature bagging        | (請用英文說明)  |  |


### 3. Feature Selection
|        ID         |        Option         |  Definiation |
|---------------------|----------------|------------------------------|
|  1 | Volcano plot  | (請用英文說明)    |  |
|  2 | Lasso regression | (請用英文說明) |   |
|  3 | Decision stump        | (請用英文說明)  |  |
|  4 | Random Forest        | (請用英文說明)  |  |
|  5 | AdaBoost        | (請用英文說明)  |  |
|  6 | Gradient boosting        | (請用英文說明)  |  |
|  7 | SVM        | (請用英文說明)  |  |


### 4. Model building


### 5. Validation


### 6. Report and visualization



# An Example for Program Demonstration    

<請描述你提供的資料內容是甚麼>

Chosse one of the following examples, double click it in jupyter interface:    
- example_feature_selection.ipynb

![image](./images/tutorial/jupyter_runall.png)
click the buttom and the script should start.

