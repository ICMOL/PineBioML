# Overview
(麻煩用中文寫三百字引言,介紹軟體目的)
This package aims to help analysis biomedical data using ML method in python.


![image](./images/workflow/auto_selection_workflow.png) 

# System Requirements
   1. Python 3.9+
   2. The following dependencies are required:
      a. numpy
      b. pandas
      c. scikit-learn
      d. matplotlib
      e. seaborn
      f. tqdm
      g. jupyter
      h. lightgbm
      i. xgboost

# Installation
### 1. Install Python 
    Please skip this step, if you have python 3.9+ installed in your PC. 

    Please follow the tutorial to install python:    
    https://learn.microsoft.com/en-us/windows/python/beginners 

    The "Visual Studio Code" and "Git" are optional.

### 2. Install Dependencies



3. ### Update modules and dependency
    1. Download release.

    2. Open Powershell (as step 2.ii), and move to the direction to the repository by entering:
        > cd direction\\to\\repository\

    3. Update pip and install dependency:
        > pip install --upgrade pip    
        > pip install -r ./requirements.txt    

    4. Open ipython interface by entering:    
        > jupyter notebook    

        This should open the browser and showing the repostory folder.    

    ![image](./images/tutorial/browser_jupyter.png)
    If the browser shows jupyter's main page, the dependency is satisfied.


# Usage example    
The fowlling operation sould be done in jupyter notebook interface within your browser (as 3.iv).    

Chosse one of the following examples, double click it in jupyter interface:    
- example_feature_selection.ipynb

![image](./images/tutorial/jupyter_runall.png)
click the buttom and the script should start.


# [Documents](./documents/main.md)
For further details.
