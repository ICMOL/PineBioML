from pandas import read_csv, read_excel, read_table


def read_file(file_path):
    """
    Sparse the file name. Support .csv, .tsv, .xlsx and R-table in .txt format.

    Args:
        file_path (str): The path to data.

    Returns:
        pandas.DataFrame: Data
    """
    # sparse csv, tsv or excel
    file_type = file_path.split(".")[-1]
    if file_type == "csv":
        file = read_csv(file_path)
    elif file_type == "tsv":
        file = read_csv(file_path, sep="\t")
    elif file_type in ["xls", "xlsx", "xlsm", "xlsb"]:
        file = read_excel(file_path)
    elif file_type == "txt":
        file = read_table(file_path)
    else:
        print(
            "Not support input type. Must be one of .csv, .tsv, .xls, .xlsx formate."
        )
        file = None
    return file
