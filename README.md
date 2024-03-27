# _______ auto Bio Feature Selection
This package aims to let biologists use ML method in python within few steps.

# Usage
1. ### Environment Setting
    If you already have python installed in your PC, then skip this step. Otherwise please follow microsoft's tutorial to install python:    
    
    https://learn.microsoft.com/zh-tw/windows/python/beginners
    
    ! You only need to follow the first step, using visual studio code is optional.

2. ### Update modules and install depqndency
    a. Download this repository and open the folder ./home/ in file explore.

    b. Open terminal/powershell, and you should see a black page with command line:
    > ps direction\\to\\repository\\home\> |

    c. Enter (or you can copy and right-click on the command line. Which will paste and execute the command.):
    > pip install --upgrade pip    
    > pip install -r ./requirements.txt

    d. Open example code by:
    > jupyter notebook    
        
    This should open the browser and showing the folder ./home.   
    If you successfully see the browser opened, then the environment setting is complete.

3. ### Execute the example:
    The fowlling operation sould be done in jupyter notebook interface within your browser.    

    a. Click and open the folder ./example     
    b. Click and open the ipython notebook package_test.ipynb    
    c. Find the button "Run all" and click it    

# Method
1. workflow
![image](./images/workflow_preprocessing.png)
![image](./images/workflow_selection.png)

    a. preprocessing
        ．impute
        ．bagging
    b. slection
    ![image](./images/workflow_base_selector.png)
        ．volvano
        ．lasso
        ．random forest
        ．decision tree
        ．SVM