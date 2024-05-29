# selection.base.normalizer
> class selection.base.normalizer(center = True, scale = True, global_scale = False)     

A preprocessing class for selection methods. For an input X it will sequantially do :
1. If center then X = X - mean(axis = 0)
2. If scale then X = X / X.std(axis = 0)
3. If global_scale then X = X / X.std(axis = [0, 1])

SVM and Lasso based methods are sensitive to the the scale of input (in numerical and in results). One of scale or global_scale sould be True, or lasso will raise an numerical error. scale will overlap the effects of global_scale.


### Parameters:

    center: boolean, default = True
        Whether to centralize in selection preprocessing. For input X, if ture then X = X- X.mean(axis = 0)

    scale: boolean, default = True
        Whether to scale after centralized. For input X, if ture then X = X / X.std(axis = 0)

    global_scale: boolean, default = False
        Whether to scale data in global. For input X, if ture then X = X / X.std(axis = [0, 1])

### Attributes:

    


# selection.base.selection
> class selection.base.selection(center = True, scale = True, global_scale = False)     

The basic pipeline for selection methods, and it include:    
1. Normalize(center = center, scale = scale, global_scale = global_scale)
2. scoring
3. choosing

parameters:

    center: boolean, default = True
        Whether to centralize in selection preprocessing. For input X, if ture then X = X- X.mean(axis = 0)

    scale: boolean, default = True
        Whether to scale after centralized. For input X, if ture then X = X / X.std(axis = 0)

    global_scale: boolean, default = False
        Whether to scale data in global. For input X, if ture then X = X / X.std(axis = [0, 1])
    



