from typing import Union
from pandas import read_csv, read_excel, read_table, concat, Series, DataFrame
from numpy import ones
import joblib
import os


def read_file(
    file_path: str,
    index_col: Union[int,
                     str] = None) -> Union[DataFrame, dict[str, DataFrame]]:
    """
    Read files, which supports common data format .csv, .tsv, .xlsx and R-table in .txt format.    
    Notice that while an excel file has several data sheet, this function will return a python dict with corresponding sheet name as dict keys.    

    Args:
        file_path (str): The path of target file.
        index_col (Union[int, str], optional): The column index or name to set as the index of dataframe. Set None to ignore. Defaults to None.

    Returns:
        Union[DataFrame, dict[str, DataFrame]]: While the target file is an excel with more than one data sheet, it will return a dict consisting of the data sheets.
    """

    # sparse csv, tsv or excel
    file_type = file_path.split(".")[-1]
    if file_type == "csv":
        file = read_csv(file_path, index_col=index_col, engine="pyarrow")
    elif file_type == "tsv":
        file_a = read_csv(file_path,
                          sep=" ",
                          index_col=index_col,
                          engine="pyarrow")
        file_b = read_csv(file_path,
                          sep="\t",
                          index_col=index_col,
                          engine="pyarrow")
        if file_a.isna().mean().mean() > file_b.isna().mean().mean():
            file = file_b
        else:
            file = file_a

    elif file_type in ["xls", "xlsx", "xlsm", "xlsb"]:
        file = read_excel(file_path, sheet_name=None, index_col=index_col)
        if len(file) == 1:
            file = file[list(file.keys())[0]]

    elif file_type == "txt":
        file = read_table(file_path, index_col=index_col, engine="pyarrow")
    else:
        raise NotImplementedError(
            "The type of target file is not supported. Must be one of .csv, .tsv, .xls, .xlsx or R table in .txt format."
        )
    return file


def read_multiple_groups(
        file_path_list: list[str],
        transpose: bool = False,
        index_col: Union[int, str] = None) -> tuple[DataFrame, Series]:
    """

    read_multiple_groups will do:    
        1. read files in    
        2. assign y according to the read-in order    
        3. concatenate all datas in row    
    It sould be used while the data was sperated into different files by their class or some kind of property we interested in.    

    Args:
        file_path_list (list[str]): A list of path to target files.
        transpose (bool, optional): Transpose before concatenating. Defaults to False.
        index_col (Union[int, str], optional): The column index or name to set as the index of dataframe. Set None to ignore. Defaults to None.

    Returns:
        tuple[DataFrame, Series]: A tuple (x, y) where x is the stacking of target files, and y is an array of integer which records each row's source in the order of file_path_list.    
    """

    # 1. read files in
    datas = []
    group_label = []
    label = 0
    for path in file_path_list:
        group = read_file(path, index_col)
        if transpose:
            group = group.T

        datas.append(group)
        # 2. assign y according to the read-in order
        group_label.append(
            Series(ones(group.shape[0]) * label, index=group.index))
        label += 1

    # 3. concatenate all the datas in row
    x = concat(datas, axis=0)
    y = concat(group_label, axis=0)

    if not len(set(y.index)) == len(y.index):
        # if index repeats, drop them all.
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
    
    y.name = 'label'

    return x, y


def save_model(model_obj: object,
               save_path: str,
               save_name: str,
               overide: bool = False) -> None:
    """
    Saving the model_obj by joblib in pickle format to the path: save_path/save_name .    
    If overide, then overide while any file with the same save_name has already existed in the save_path.

    Args:
        model_obj (object): The model or pipeline to be saved.
        save_path (str): The path to to save the model_obj.
        save_name (str): the saving filename.
        overide (bool, optional): True for overidind any file with save_name in save_path. Defaults to False.
    """
    if not save_path[-1] == "/":
        save_path = save_path + "/"

    if not os.path.exists(save_path):
        print(save_path, " does not exist yet. we will try to create it.")
        os.makedirs(save_path)

    if os.path.exists(save_path + save_name):
        print(save_name, " has already exist in ", save_path)
        if overide:
            print("It will be overide.")
            joblib.dump(model_obj, save_path + save_name)
        else:
            print(
                "please choose another model save_name or set overide to True which will replace the existing one"
            )
    else:
        joblib.dump(model_obj, save_path + save_name)


def load_model(save_path: str, save_name: str = None) -> object:
    """
    Load a saved model.    

    If save_name was provided, load_model will try to access save_path/save_name .    
    else save_path will be regarded as save_path/save_name (lazy mode).    

    For example, a model saved as "./output/model/best_model"    
        load_model("./output/model/", "best_model")    
        load_model("./output/model/best_model")    
    have exact the same result.

    Args:
        save_path (str): The path to folder where the model saved.
        save_name (str, optional): The name of target model. Defaults to None.

    Returns:
        object: An python object that was saved in previous. DO NOT TRUST UNKNOWN SOURCE PICKLE FILES.
    """
    if not save_name is None:
        if not save_path[-1] == "/":
            save_path = save_path + "/"

        return joblib.load(save_path + save_name)
    else:
        # lazy mode
        return joblib.load(save_path)
