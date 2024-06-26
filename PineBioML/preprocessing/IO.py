from pandas import read_csv, read_excel, read_table, concat, Series
from numpy import ones


def read_file(file_path, index_col=0):
    """
    Sparse the file name. Support .csv, .tsv, .xlsx and R-table in .txt format.

    Args:
        file_path (str): The path to data.
        index_col (int or str): The column to set as the index of dataframe

    Returns:
        pandas.DataFrame: Data
    """
    # sparse csv, tsv or excel
    file_type = file_path.split(".")[-1]
    if file_type == "csv":
        file = read_csv(file_path, index_col=index_col)
    elif file_type == "tsv":
        file = read_csv(file_path, sep="\t", index_col=index_col)
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
    files = []
    group_label = []
    label = 0
    for path in file_path_list:
        data = read_file(path, index_col)
        if transpose:
            data = data.T

        files.append(data)
        group_label.append(
            Series(ones(data.shape[0]) * label, index=data.index))
        label += 1

    return concat(files, axis=0), concat(group_label, axis=0)
