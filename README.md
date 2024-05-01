# _______ auto Bio-Feature Selection
This package aims to help you analysis bio data in tabular format using ML method in python. In the First step, we will provide feature selection and autoML.

# Index
1. Installation
2. Usage
3. Method


### !! WARNING: It's a pre-released version, which is unstable and lack of testing. Wellcome to be the first user.
### If you meet any problem or have any idea, just raise an issue or contact author.

# Installation
0. ### Requirement
    ．OS: windows10 or 11

1. ### Environment Setting
    If you already have python >= 3.7 installed in your PC, then skip to step2. Otherwise please follow microsoft's tutorial to install python:    
    ! You only need to follow the first step in the turorial, wheather to use visual studio code is optional.
    https://learn.microsoft.com/en-us/windows/python/beginners
    
    

2. ### Update modules and install depqndency
    a. Download release.

    b. Open terminal/powershell, and you should see a black page with command line:
    > ps direction\\to\\repository\\home\> |

    c. Entering (or you can copy and right-click on the command line. Which will paste and execute the command.):
    > pip install --upgrade pip    
    > pip install -r ./requirements.txt

    d. Open ipython interface by entering:
    > jupyter notebook    
        
    This should open the browser and showing the folder ./home.   
    If you successfully see the browser opened, then the environment setting is complete.

3. ### Run the example:
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