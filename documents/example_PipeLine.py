#!/usr/bin/env python
# coding: utf-8

# # IO

# In[1]:


import PineBioML.IO as IO
import sys


# In[2]:


# Commands
input_folder = sys.argv[1] # path to input folder, ex "./input/"
input_filename = sys.argv[2] # input file name, ex "HAPTdata.csv"
# the program will try to read input_folder + input_filename, ex "./input/HAPTdata.csv"

index_column = int(sys.argv[3]) # the order of index column, if there is no index in data then set it to -1, ex "-1"
if index_column == -1:
    index_column = None
target_column_name = sys.argv[4] # literally the column name of target. ex "target"

output_folder = "./output/"
export_title = "example_basic "


# # example_PipeLine.py ./input/ example_data.xlsx 0 y
# 
# data = IO.read_file(input_folder+ "example_data.xlsx", index_col=0)
# 
# ##### pick the target label column
# y = data["y"]
# 
# ##### drop the target from dataframe, using the remaining to predict y (the target)
# x = data.drop("y", axis = 1)

# In[3]:


# example_PipeLine.py ./input/ HAPTdata.csv -1 target

data = IO.read_file(input_folder+ input_filename, index_col=index_column)

##### pick the target label column
y = data[target_column_name]

##### drop the target from dataframe, using the remaining to predict y (the target)
x = data.drop(target_column_name, axis = 1)


# # Train test split

# In[4]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.33, random_state=142, shuffle=True)


# # Data Overview

# In[5]:


from PineBioML.report.utils import data_overview
data_overview(x_train, y_train, show_fig = False)


# # Modeling

# In[6]:


from PineBioML.preprocessing import Normalizer
from PineBioML.preprocessing import impute
from PineBioML.selection.ensemble import selector
import PineBioML.model.supervised.Classification as tuner

from sklearn.pipeline import Pipeline


# In[7]:


pipe = Pipeline(
    [
        ("Standarization", Normalizer(method="StandardScaler")),
        ("Missing_Value_Processing", impute.simple_imputer(threshold = 0.6, strategy = "constant")),
        #("Feature_Selection", selector(z_importance_threshold = 1)),
        ("Modeling", tuner.RandomForest_tuner(target="qwk"))
     ]
)


# In[8]:


pipe.fit(x_train, y_train)


# # Evaluation

# In[9]:


from PineBioML.report.utils import classification_summary

classification_summary(y_valid, pipe.predict_proba(x_valid), show_fig=False)


# # Saving Model

# In[10]:


# save model
IO.save_model(pipe, output_folder+"models/", export_title+"MostInOne", overide=True)


# # Reload model

# # load model
# presaved_model = IO.load_model(output_folder+"models/", export_title+"MostInOne")

# # prediction
# presaved_model.predict(x_valid)

# classification_summary(y_valid, presaved_model.predict_proba(x_valid)[:, 1]) 
