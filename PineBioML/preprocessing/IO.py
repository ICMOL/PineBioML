from pandas import read_csv, read_excel, read_table, concat, Series
from numpy import ones


def read_file(file_path, index_col=0):
    """
    Read files, which supports common data format .csv, .tsv, .xlsx and R-table in .txt format.    
    Notice that while an excel file has several data sheet, this function will return a python dict with corresponding sheet name as dict keys.    

    Args:
        file_path (str): The path to data.
        index_col (int or str): Default is 0. The column index or name to set as the index of dataframe. Set None to ignore.

    Returns:
        pandas.DataFrame: Data or a dict to data sheets
    """
    # sparse csv, tsv or excel
    file_type = file_path.split(".")[-1]
    if file_type == "csv":
        file = read_csv(file_path, index_col=index_col)
    elif file_type == "tsv":
        file = read_csv(file_path, sep=" ", index_col=index_col)
    elif file_type in ["xls", "xlsx", "xlsm", "xlsb"]:
        file = read_excel(file_path, sheet_name=None, index_col=index_col)
        if len(file) == 1:
            file = file[list(file.keys())[0]]

    elif file_type == "txt":
        file = read_table(file_path, index_col=index_col)
    else:
        print(
            "Not support input type. Must be one of .csv, .tsv, .xls, .xlsx formate."
        )
        file = None
    return file


def read_multiple_files(file_path_list, transpose=False, index_col=0):
    """
    Read files in file_path_list

    Args:
        file_path_list (list): List of string. The pathes to data.
        transpose (boolean): weather to transpose input.

    Returns:
        list: a list contains data in pandas.DataFrame formate.
    """
    # sparse csv, tsv or excel
    datas = []
    group_label = []
    label = 0
    for path in file_path_list:
        group = read_file(path, index_col)
        if transpose:
            group = group.T

        datas.append(group)
        group_label.append(
            Series(ones(group.shape[0]) * label, index=group.index))
        label += 1

    return concat(datas, axis=0), concat(group_label, axis=0)
