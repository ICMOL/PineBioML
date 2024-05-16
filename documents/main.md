# Workflow
![image](../images/workflow/auto_selection_workflow.png)

### Input format
- Input data sould be tabular and put in the ./input folder.
- We support .csv/.tsv/.xlsx as input.
- *Sample ID is required. You can assign a column name as indexing ID. If ID of x and y do not match, an error will raise.

### Missing value
We provied several method to dealing with missing values:
1. Drop    
    Drop those who has high missing rate. Defaut threshold is 33%.
2. Constant impute 
    Impute missing value in a column by a single value such as 0, column mean or column quantiles
3. KNN impute
    Impute missing value by the mean or quantiles of k nearest sample.
4. *mice impute
    [mice](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/)
5. *modeling impute
    Impute as self-regression.

### Customize Transform
1. PCA    
    Reduce dimension
2. Power transform    
    To deal with heavy tail problem. Some statistical method requires high normality, then you may use a cox-box transform or yoe-johnson transform to data.
3. Feature Bagging     
    Bagging simalar features according to their covariance or feature name similarity. For example: AAC32 AAC1 YUFGK6 => AAC* YUFGK

### Feature selection
We provide several method to scoring how important a feature is. The base pipeline is normalize->scoring->choosing. Some method is sensitive or requiring a spetial data scale, and these will be adapted for each method. Scoring is the core of selecting methods such as *f-score, p-value + fold change, and several model based methods. choosing is basically ranking and pikcing.

### Modeling

