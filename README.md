# _______ auto Bio-Feature Selection
### Overview
![image](./images/workflow/auto_selection_workflow.png) 
This package aims to help you analysis bio data in tabular format using ML method in python.   

In the First stage, we will provide feature selection and autoML.

(Somemore description or some stuff)

# Index
- Installation
    - Requirement
    - Python Installation
    
- Usage example
    - Feature selection
    - auto ml
- [Documents](./documents/main.md)


# Installation
0. ### Requirement
    - Windows10 or 11
    - Python >= 3.9
    - dependency in requirements.txt

1. ### Python Installation
    If you already have python >= 3.9 installed in your PC, then skip to next step *Powershell and Python*.   

    Please follow microsoft's tutorial to install python:    
    https://learn.microsoft.com/en-us/windows/python/beginners    
    You only need to finish the first part *Install Python* in the turorial, wheather to use visual studio code is optional. In this project, we use ipython notebook mainly. 

2. ### Powershell and Python
    1. Open the Start menu, searching for Windows PowerShell.    
        ![image](./images/tutorial/open_powershell.png)
    2. It will open a terminal like this:
        ![image](./images/tutorial/powershell_window.png)    
    3. Type in python.    
        ![image](./images/tutorial/powershell_python.png)    
    4. You should see the version information of python that you just installed. 
        ![image](./images/tutorial/powershell_python_result.png)
    5. In final to exit python terminal, type in exit()
        ![image](./images/tutorial/powershell_exit.png)



3. ### Update modules and dependency
    1. Download release.

    2. Open Powershell (as step 2.2), and move to the direction to the repository by entering:
        > cd direction\\to\\repository\

    3. Update pip and install dependency:
        > pip install --upgrade pip    
        > pip install -r ./requirements.txt    

    4. Open ipython interface by entering:    
        > jupyter notebook    

        This should open the browser and showing the repostory folder.    

    If the browser is opened, then the dependency is required.
    ![image](./images/tutorial/browser_jupyter.png)


# Usage example    
The fowlling operation sould be done in jupyter notebook interface within your browser (as 3.4).    

Try one of the following examples:    
- example_feature_selection.ipynb

