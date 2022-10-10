import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import joblib
import pickle
import datetime
import warnings
import json
import os
import io
import shutil
#import sys


def get_path_unit_label(path):
    """
    Classifies a path by its storage unit type.

    :param path: storage unit path
    :return: "directory" for directory, "file" for file or symbolic link, "" otherwise
    """
    try:
        if os.path.isdir(path):
            return "directory"
        elif os.path.isfile(path) or os.path.islink(path):
            return "file"
        else:
            return ""
    except TypeError:
        return ""


def remove_path_unit(path):
    """
    Deletes storage unit (directory or file) for given path if it exists.

    :param path: storage unit path
    """
    path_label = get_path_unit_label(path)
    if path_label == "directory":
        shutil.rmtree(path)
    elif path_label == "file":
        os.unlink(path)


def empty_directory(directory):
    """
    Deletes contents from selected directory if it exists.

    :param directory: directory to empty
    """
    if get_path_unit_label(directory) == "directory":
        for unit in os.listdir(directory):
            remove_path_unit(join_path(directory, unit))


def make_directory(directory):
    """
    Makes selected directory if it doesn't exist.

    :param directory: directory to make
    """
    if get_path_unit_label(directory) != "directory":
        os.makedirs(directory)


def get_data_label(data):
    """
    Classifies an object by its data type.

    :param data: data object
    :return: "list" for list,
             "tuple" for tuple,
             "set" for set,
             "dict" for dictionary,
             "ndarray" for NumPy ndarray,
             "df" for pandas DataFrame,
             "series" for pandas Series,
             "index" for pandas Index,
             "datetime" for datetime datetime or pandas Timestamp,
             "date" for datetime date,
             "time" for datetime time,
             "timedelta" for datetime timedelta,
             "str" for string (including NumPy),
             "bool" for boolean,
             "int" for integer (including NumPy),
             "float" for floating point (including NumPy),
             "complex" for complex (including NumPy),
             "sklearn" for any scikit-learn object,
             "fig" for matplotlib Figure,
             "map" for Folium Map
             "textio" for TextIOBase,
             "bufferedio" for BufferedIOBase,
             "rawio" for RawIOBase,
             "" otherwise
    """
    if isinstance(data, list):
        return "list"
    elif isinstance(data, tuple):
        return "tuple"
    elif isinstance(data, set):
        return "set"
    elif isinstance(data, dict):
        return "dict"
    elif isinstance(data, np.ndarray):
        return "ndarray"
    elif isinstance(data, pd.DataFrame):
        return "df"
    elif isinstance(data, pd.Series):
        return "series"
    elif isinstance(data, pd.Index):
        return "index"
    elif (type(data) is datetime.datetime) or (type(data) is pd.Timestamp):
        return "datetime"
    elif type(data) is datetime.date:
        return "date"
    elif type(data) is datetime.time:
        return "time"
    elif type(data) is datetime.timedelta:
        return "timedelta"
    elif isinstance(data, (str, np.string_)):
        return "str"
    elif isinstance(data, (bool, np.bool_)):
        return "bool"
    elif np.issubdtype(type(data), np.integer):
        return "int"
    elif np.issubdtype(type(data), np.floating):
        return "float"
    elif np.issubdtype(type(data), np.complex_):
        return "complex"
    elif "sklearn" in str(type(data)):
        return "sklearn"
    elif "tslearn" in str(type(data)):
        return "tslearn"
    elif isinstance(data, plt.Figure):
        return "fig"
    elif isinstance(data, folium.Map):
        return "map"
    elif isinstance(data, io.TextIOBase):
        return "textio"
    elif isinstance(data, io.BufferedIOBase):
        return "bufferedio"
    elif isinstance(data, io.RawIOBase):
        return "rawio"
    else:
        return ""


def convert_args_to_strings(*args, **formats):
    """
    Converts arguments to strings.

    :param args: variable number of arguments
    :param formats: variable number of keyword arguments for conversion formats, default values in () -
                    "datetime" for datetime datetime or pandas Timestamp (%Y%m%d%H%M),
                    "date" for datetime date (%Y%m%d),
                    "time" for datetime time (%H%M),
                    "unsupported" for unsupported arguments ("")
    :return: string arguments list
    """
    # formats from kwargs or default
    format_datetime = formats.get("datetime", "%Y%m%d%H%M")
    format_date = formats.get("date", "%Y%m%d")
    format_time = formats.get("time", "%H%M")
    format_unsupported = formats.get("unsupported", "")

    str_args = []
    for arg in args:
        arg_type = get_data_label(arg)

        if arg_type != "str":
            if arg_type == "datetime":
                arg = arg.strftime(format_datetime)
            elif arg_type == "date":
                arg = arg.strftime(format_date)
            elif arg_type == "time":
                arg = arg.strftime(format_time)
            elif arg_type == "list":
                arg = convert_args_to_strings(*arg, **formats)
            elif arg_type in {"tuple", "set"}:
                arg = convert_args_to_strings(*(list(arg)), **formats)
            elif arg_type in {"ndarray", "index"}:
                arg = convert_args_to_strings(*(arg.tolist()), **formats)
            elif arg_type in {"bool", "int", "float", "complex"}:
                arg = str(arg)
            else:
                arg = format_unsupported

        str_args.append(arg)

    return str_args


def join_file(*parts, **options):
    """
    Joins file parts together.

    :param parts: file parts to join together
    :param options: variable number of keyword arguments for options, default values in () -
                    "sep" string to add separator of file parts (""),
                    "ext" string to add file extension (""),
                    "convert" boolean to convert parts to strings before joining (False),
                    for conversion formats see convert_args_to_strings documentation for options
    :return: joined file or "" if joining failed
    """
    # options from kwargs or default
    option_sep = options.pop("sep", "")
    option_ext = options.pop("ext", "")
    option_convert = options.pop("convert", False)

    parts = [x for x in parts if pd.notna(x)]  # remove nan

    if option_convert:
        parts = convert_args_to_strings(*parts, **options)

    try:
        file = option_sep.join(parts)
    # not all parts are strings
    except TypeError:
        return ""

    if option_ext != "":
        if not option_ext.startswith("."):
            option_ext = "." + option_ext
        file += option_ext

    return file


def join_path(*parts, **options):
    """
    Joins path parts together.

    :param parts: path parts to join together
    :param options: variable number of keyword arguments for options, default values in () -
                    "cwd" boolean to add current working directory at the start of the path (False),
                    "cwdp" boolean to add current working directory's parent directory at the start of the path (False),
                    "root" boolean to add project root (utils.py location) at the start of path (False),
                    "osroot" boolean to add filesystem root at the start of path (False),
                    "norm" boolean to normalize joined path (True),
                    "convert" boolean to convert parts to strings before joining (False),
                    for conversion formats see convert_args_to_strings documentation for options
    :return: joined path or "" if joining failed
    """
    # options from kwargs or default
    option_cwd = options.pop("cwd", False)
    option_cwdp = options.pop("cwdp", False)
    option_root = options.pop("root", False)
    option_osroot = options.pop("osroot", False)
    option_norm = options.pop("norm", True)
    option_convert = options.pop("convert", False)

    if option_convert:
        parts = convert_args_to_strings(*parts, **options)

    try:
        if option_cwd:
            path = os.path.join(os.getcwd(), *parts)
        elif option_cwdp:
            path = os.path.join(os.path.abspath(os.pardir), *parts)
        elif option_root:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)
        elif option_osroot:
            path = os.path.join(os.sep, os.path.abspath(os.sep), *parts)
        else:
            path = os.path.join(*parts)
    # not all parts are strings
    except TypeError:
        return ""

    if option_norm:
        return os.path.normpath(path)
    else:
        return path


def join_directory_and_file_path(file, directory="", make=False):
    """
    Joins directory and file paths together if joined path is valid.

    :param file: file name or absolute path
    :param directory: file directory, if "" use root project directory, ignored if file parameter is absolute path
    :param make: make selected directory if it doesn't exist and don't check if file exists if True
    :return: joined path if make parameter is False and file exists or if make parameter is True and directory exists,
             "" otherwise (if file parameter is an absolute path then it's split into directory and file for validation)
    """
    try:
        if os.path.isabs(file):
            directory, file = os.path.split(file)
    # invalid file
    except TypeError:
        return ""

    if directory != "":
        if make:
            try:
                make_directory(directory)
            # invalid directory path, failed to make
            except OSError:
                return ""
        # invalid directory path
        elif get_path_unit_label(directory) != "directory":
            return ""

        file_path = join_path(directory, file)
    else:
        file_path = join_path(file, root=True)

    # invalid file path
    if not make and get_path_unit_label(file_path) != "file":
        return ""

    return file_path


def read_data_from_file(file, directory="", reading="", **options):
    """
    Reads data from file.

    :param file: file name or absolute path
    :param directory: file directory, if "" use current working directory
    :param reading: data structure being read from file -
                    list if "list",
                    dictionary if "dict", "json", or "JSON",
                    unpickled object if "pickle",
                    unpickled joblib object/model if "joblib" or "sklearn",
                    tslearn model if "tslearn",
                    DataFrame if "df", "dataframe", "DataFrame", "csv", or "CSV",
                    ndarray if "ndarray" or "array",
                    image if "image", "png", "PNG", "jpg", or "JPG",
                    string if "string" or "str",
                    infer parameter if "" (DataFrame for .csv or .CSV files, dictionary for ".json" or ".JSON",
                    unpickled object for ".pickle", unpickled joblib object for ".joblib" or ".sklearn", image for
                    ".png", ".PNG", ".jpg", ".JPG", ".jpeg", or ".JPEG", string for ".html", otherwise list)
    :param options: variable number of keyword arguments for options, default values in () -
                    if reading a list -
                    "end" string to separate items in file (newline),
                    "sep" string to separate values in items for 2D lists (" "),
                    "lstrip" string characters to strip from the left of each item (" "),
                    "rstrip" string characters to strip from the right of each item (" "),
                    "empty" boolean to add empty items to data (False),
                    "dim2" boolean to make each item a list, even if singular (False),
                    "encoding" string to choose encoding for reading ("utf-8")
                    if reading a dictionary -
                    "loads" boolean to use json.loads on read file instead of using json.load (False),
                    "encoding" string to choose encoding for reading ("utf-8"),
                    see json.load and json.loads documentation for additional options
                    if reading a pickled object -
                    "encoding" string to choose encoding for reading ("utf-8"),
                    see pickle.load documentation for additional options
                    if reading a joblib pickled object see joblib.load documentation for options
                    if reading a tslearn model -
                    "type" string to select file type, "pickle" for pickled, "json" for JSON, "hdf5" for HDF5 if h5py is
                    installed ("pickle")
                    "mclass" object to select object class read from file (None)
                    if reading a DataFrame see pandas.read_csv documentation for options
                    if reading an ndarray see numpy.loadtxt documentation for options
                    if readin an image see matplotlib.pyplot.imread for options
    :return: file data or None if reading failed
    """
    file_path = join_directory_and_file_path(file, directory)
    if file_path == "":
        return None

    # infer data structure
    if reading == "":
        try:
            ext = os.path.splitext(file)[1]
        except TypeError:
            return None
        if ext in {".csv", ".CSV", ".json", ".JSON", ".pickle", ".joblib"}:
            reading = ext[1:]
        elif ext == ".sklearn":
            reading = "joblib"
        elif ext in {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}:
            reading = "image"
        elif ext == ".html":
            reading = "string"
        else:
            reading = "list"

    # input data
    if reading == "list":
        option_end = options.pop("end", "\n")
        option_sep = options.pop("sep", " ")
        option_lstrip = options.pop("lstrip", " ")
        option_rstrip = options.pop("rstrip", " ")
        option_empty = options.pop("empty", False)
        option_dim2 = options.pop("dim2", False)
        option_encoding = options.pop("encoding", "utf-8")

        data = []
        with open(file_path, "r", encoding=option_encoding) as file_in:
            in_data = file_in.read()

            # separate items
            if option_end != "":
                in_data = in_data.split(option_end)
            else:
                in_data = list(in_data)

            for item in in_data:
                # skip empty
                if not option_empty and item == "":
                    continue

                # strip
                if option_lstrip != "":
                    item.lstrip(option_lstrip)
                if option_rstrip != "":
                    item.rstrip(option_rstrip)

                # separate values in item if multiple
                if option_sep != "":
                    item = item.split(option_sep)
                else:
                    item = list(item)

                if len(item) == 1 and not option_dim2:
                    data.append(item[0])
                else:
                    data.append(item)
    elif reading in {"dict", "json", "JSON"}:
        option_loads = options.pop("loads", False)
        option_encoding = options.pop("encoding", "utf-8")

        with open(file_path, "r", encoding=option_encoding) as file_in:
            if option_loads:
                data = json.loads(file_in.read(), **options)
            else:
                data = json.load(file_in, **options)
    elif reading == "pickle":
        option_encoding = options.pop("encoding", "utf-8")

        with open(file_path, "r", encoding=option_encoding) as file_in:
            data = pickle.load(file_in, **options)
    elif reading in {"joblib", "sklearn"}:
        data = joblib.load(file_path, **options)
    elif reading == "tslearn":
        option_type = options.get("type", "pickle")
        option_mclass = options.get("mclass", None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)  # h5py not installed warning
            if option_type == "pickle":
                data = option_mclass.from_pickle(file_path)
            elif option_type == "json":
                data = option_mclass.from_json(file_path)
            elif option_type == "hdf5":
                data = option_mclass.from_hdf5(file_path)
    elif reading in {"df", "dataframe", "DataFrame", "csv", "CSV"}:
        data = pd.read_csv(file_path, **options)
    elif reading in {"ndarray", "array"}:
        data = np.loadtxt(file_path, **options)
    elif reading in {"image", "png", "PNG", "jpg", "JPG"}:
        data = plt.imread(file_path, **options)
    elif reading in {"str", "string"}:
        option_encoding = options.pop("encoding", "utf-8")

        with open(file_path, "r", encoding=option_encoding) as file_in:
            data = file_in.read()
    else:
        return None

    return data


def write_data_to_file(data, file, directory="", **options):
    """
    Writes data to file.

    :param data: file data as list (1D or 2D), dictionary, sklearn model, tslearn model, DataFrame, Series, ndarray,
                 figure, or map, otherwise handled as list if data type is supported
    :param file: file name or absolute path
    :param directory: file directory, if "" use current working directory
    :param options: variable number of keyword arguments for options, default values in () -
                    if writing a list (or converted list) -
                    "end" string to separate items in file (newline),
                    "sep" string to separate values in items for 2D lists (" "),
                    "lstrip" string characters to strip from the left of each item (" "),
                    "rstrip" string characters to strip from the right of each item (" "),
                    "convert" boolean to convert list items to strings (False),
                    "encoding" string to choose encoding for writing ("utf-8")
                    if writing a dictionary -
                    "dumps" boolean to use json.dumps to write a string instead of using json.dump (False),
                    "encoding" string to choose encoding for writing ("utf-8"),
                    see json.dump and json.dumps documentation for options
                    if writing an sklearn model see joblib.dump documentation for options
                    if writing a tslearn model -
                    "type" string to select file type, "pickle" for pickled, "json" for JSON, "hdf5" for HDF5 if h5py is
                    installed ("pickle")
                    if writing a DataFrame see pandas.DataFrame.to_csv documentation for options
                    if writing a Series see pandas.Series.to_csv documentation for options
                    if writing a ndarray see numpy.savetxt documentation for options
                    if writing a Figure see matplotlib.pyplot.savefig documentation for options
                    if writing a Map see folium.Map.save documentation for options
    """
    file_path = join_directory_and_file_path(file, directory, make=True)
    if file_path == "":
        return

    writing = get_data_label(data)
    # not supported
    if writing in {"", "textio", "bufferedio", "rawio"}:
        return

    try:
        # handle other data types as list
        if writing not in {"list", "dict", "sklearn", "tslearn", "df", "series", "ndarray", "fig", "map"}:
            data = [data]
            writing = "list"

        # output data
        if writing == "list":
            # options from kwargs or default
            option_end = options.pop("end", "\n")
            option_sep = options.pop("sep", " ")
            option_lstrip = options.pop("lstrip", " ")
            option_rstrip = options.pop("rstrip", " ")
            option_convert = options.pop("convert", False)
            option_encoding = options.pop("encoding", "utf-8")

            if option_convert:
                data = convert_args_to_strings(*data, **options)

            with open(file_path, "w", encoding=option_encoding) as file_out:
                for item in data:
                    item_type = get_data_label(item)
                    # join values of item if multiple
                    if item_type == "list":
                        try:
                            item = option_sep.join([value for value in item])
                        # not all values in sublist are strings, skip item
                        except TypeError:
                            continue
                    # not string, skip item
                    elif item_type != "str":
                        continue

                    if option_lstrip != "":
                        item = item.lstrip(option_lstrip)
                    if option_rstrip != "":
                        item = item.rstrip(option_rstrip)

                    # item separator
                    if option_end != "":
                        item += option_end

                    file_out.write(item)
        elif writing == "dict":
            option_dumps = options.pop("dumps", False)
            option_encoding = options.pop("encoding", "utf-8")

            with open(file_path, "w", encoding=option_encoding) as file_out:
                if option_dumps:
                    file_out.write(json.dumps(data, **options))
                else:
                    json.dump(data, file_out)
        elif writing == "sklearn":
            joblib.dump(data, file_path, **options)
        elif writing == "tslearn":
            option_type = options.get("type", "pickle")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)  # h5py not installed warning
                if option_type == "pickle":
                    data.to_pickle(file_path)
                elif option_type == "json":
                    data.to_json(file_path)
                elif option_type == "hdf5":
                    remove_path_unit(file_path)  # avoid FileExistsError
                    data.to_hdf5(file_path)
        elif writing in {"df", "series"}:
            data.to_csv(file_path, **options)
        elif writing == "ndarray":
            np.savetxt(file_path, data, **options)
        elif writing == "fig":
            data.savefig(file_path, **options)
        elif writing == "map":
            data.save(file_path, **options)
    except Exception:
        return


def read_config_options(networks_config=False, stations_config=False, datetimes_config=False):
    """
    Reads selected options from config files.

    :param networks_config: get network list from network config file if True
    :param stations_config: get station list from station config file if True
    :param datetimes_config: get datetime list with Timestamp objects from datetimes config file if True
    :return: config options dictionary of lists or config options list if only 1 parameter is True, with None where
             reading failed
    """
    config_options = dict()
    if networks_config:
        networks = read_data_from_file("networks.txt", join_path("config", root=True))
        if networks is None:
            config_options["networks"] = None
        else:
            config_options["networks"] = [network.upper() for network in networks]

    if stations_config:
        stations = read_data_from_file("stations.txt", join_path("config", root=True))
        if stations is None:
            config_options["stations"] = None
        else:
            config_options["stations"] = [station.upper() for station in stations]

    if datetimes_config:
        dates = read_data_from_file("datetimes.txt", join_path("config", root=True))
        if dates is None:
            config_options["datetimes"] = None
        else:
            config_options["datetimes"] = pd.to_datetime(dates)

    if len(config_options) == 1:
        return list(config_options.items())[0][1]
    else:
        return config_options


def read_network_info(networks=None, main_columns=False):
    """
    Reads network info from file.

    :param networks: network code or network code list to select info from networks
    :param main_columns: include only network code column if True, otherwise include all columns
    :return: network info DataFrame or None if reading failed
    """
    network_info = read_data_from_file("networks.csv", join_path("data", root=True))
    if network_info is None:
        return None

    # keep selected networks
    if networks is not None:
        if get_data_label(networks) == "str":
            networks = [networks]
        network_info = network_info[network_info["network"].isin(set(networks))]

    # keep selected columns
    if main_columns:
        select_columns = {"network"}
        network_info = network_info[network_info.columns[network_info.columns.isin(select_columns)]]

    return network_info


def write_network_info(network_info, networks=None, main_columns=False, index_column=False):
    """
    Writes network info to file.

    :param network_info: network info DataFrame
    :param networks: network code or network code list to select info from networks
    :param main_columns: include only network code column if True, otherwise include all columns
    :param index_column: include index column if True
    """
    # keep selected networks
    if networks is not None:
        if get_data_label(networks) == "str":
            networks = [networks]
        network_info = network_info[network_info["network"].isin(set(networks))]

    # keep selected columns
    if main_columns:
        select_columns = {"network"}
        network_info = network_info[network_info.columns[network_info.columns.isin(select_columns)]]

    write_data_to_file(network_info, "networks.csv", join_path("data", root=True), index=index_column)


def read_station_info(networks=None, stations=None, main_columns=False, start_datetime=None, end_datetime=None):
    """
    Reads station info from file.

    :param networks: network code or network code list to select info from networks
    :param stations: station ID or station ID list to select info from stations
    :param main_columns: include only station ID and network code columns if True, otherwise include all columns
    :param start_datetime: start datetime for info as Timestamp object or None to use the minimum pandas Timestamp
                           (ignored if start_datetime and end_datetime parameters are None)
    :param end_datetime: end datetime for info as Timestamp object or None to use the maximum pandas Timestamp
                         (ignored if start_datetime and end_datetime parameters are None)
    :return: station info DataFrame or None if reading failed
    """
    station_info = read_data_from_file("stations.csv", join_path("data", root=True))
    if station_info is None:
        return None

    # keep selected networks
    if networks is not None:
        if get_data_label(networks) == "str":
            networks = [networks]
        station_info = station_info[station_info["network"].isin(set(networks))]

    # keep selected stations
    if stations is not None:
        if get_data_label(stations) == "str":
            stations = [stations]
        station_info = station_info[station_info["sid"].isin(set(stations))]

    # format datetime columns
    columns = set(station_info.columns)
    if "start" in columns:
        station_info["start"] = pd.to_datetime(station_info["start"], format="%Y")
    if "end" in columns:
        station_info["end"] = pd.to_datetime(station_info["end"], format="%Y")

    # keep stations in time range
    if start_datetime is not None or end_datetime is not None:
        if start_datetime is None:
            start_datetime = pd.Timestamp.min
        if end_datetime is None:
            end_datetime = pd.Timestamp.max
        if not end_datetime < start_datetime:
            # start in time period, end in time period, or are ongoing in time period
            if station_info is not None:
                station_info["endf"] = station_info["end"].fillna(pd.Timestamp.now())  # stations still in service
                station_info = station_info[((station_info["start"] >= start_datetime) &
                                             (station_info["start"] <= end_datetime)) |
                                            ((station_info["endf"] >= start_datetime) &
                                             (station_info["endf"] <= end_datetime)) |
                                            ((station_info["start"] < start_datetime) &
                                             (station_info["endf"] > end_datetime))]
                station_info = station_info.drop("endf", axis=1)

    # keep selected columns
    if main_columns:
        select_columns = {"sid", "network"}
        station_info = station_info[station_info.columns[station_info.columns.isin(select_columns)]]

    return station_info


def write_station_info(station_info, networks=None, stations=None, main_columns=False, index_column=False,
                       start_datetime=None, end_datetime=None):
    """
    Writes station info to file.

    :param station_info: station info DataFrame
    :param networks: network code or network code list to select info from networks
    :param stations: station ID or station ID list to select info from stations
    :param main_columns: include only station ID and network code columns if True, otherwise include all columns
    :param index_column: include index column if True
    :param start_datetime: start datetime for info as Timestamp object or None to use the minimum pandas Timestamp
                           (ignored if start_datetime and end_datetime parameters are None)
    :param end_datetime: end datetime for info as Timestamp object or None to use the maximum pandas Timestamp
                         (ignored if start_datetime and end_datetime parameters are None)
    """
    # keep selected networks
    if networks is not None:
        if get_data_label(networks) == "str":
            networks = [networks]
        station_info = station_info[station_info["network"].isin(set(networks))]

    # keep selected stations
    if stations is not None:
        if get_data_label(stations) == "str":
            stations = [stations]
        station_info = station_info[station_info["sid"].isin(set(stations))]

    # keep stations in time range
    if start_datetime is not None or end_datetime is not None:
        if start_datetime is None:
            start_datetime = pd.Timestamp.min
        if end_datetime is None:
            end_datetime = pd.Timestamp.max
        if not end_datetime < start_datetime:
            # start in time period, end in time period, or are ongoing in time period
            if station_info is not None:
                station_info["endf"] = station_info["end"].fillna(pd.Timestamp.now())  # stations still in service
                station_info = station_info[((station_info["start"] >= start_datetime) &
                                             (station_info["start"] <= end_datetime)) |
                                            ((station_info["endf"] >= start_datetime) &
                                             (station_info["endf"] <= end_datetime)) |
                                            ((station_info["start"] < start_datetime) &
                                             (station_info["endf"] > end_datetime))]
                station_info = station_info.drop("endf", axis=1)

    # keep selected columns
    if main_columns:
        select_columns = {"sid", "network"}
        station_info = station_info[station_info.columns[station_info.columns.isin(select_columns)]]

    write_data_to_file(station_info, "stations.csv", join_path("data", root=True), index=index_column, date_format="%Y")


def read_fire_info(networks=None, main_columns=False, start_datetime=None, end_datetime=None, min_spread=0):
    """
    Reads selected fire info from file.

    :param networks: network code or network code list to select info from networks
    :param main_columns: include only network code, latitude, longitude, spread, start date and end date columns if
                         True, otherwise include all columns
    :param start_datetime: start datetime for info as Timestamp object or None to use the minimum pandas Timestamp
                           (ignored if start_datetime and end_datetime parameters are None)
    :param end_datetime: end datetime for info as Timestamp object or None to use the maximum pandas Timestamp
                         (ignored if start_datetime and end_datetime parameters are None)
    :param min_spread: minimum fire spread in hectares for info
    :return: fire info DataFrame or None if reading failed
    """
    fire_info = read_data_from_file("fires.csv", join_path("data", root=True))
    if fire_info is None:
        return None

    # keep selected networks
    if networks is not None:
        if get_data_label(networks) == "str":
            networks = [networks]
        fire_info = fire_info[fire_info["network"].isin(set(networks))]

    # keep selected columns
    if main_columns:
        select_columns = {"network", "lat", "lon", "spread", "start", "end"}
        fire_info = fire_info[fire_info.columns[fire_info.columns.isin(select_columns)]]

    # format datetime columns
    columns = set(fire_info.columns)
    if "start" in columns:
        fire_info["start"] = pd.to_datetime(fire_info["start"])
    if "end" in columns:
        fire_info["end"] = pd.to_datetime(fire_info["end"])

    # keep fires in time range
    if start_datetime is not None or end_datetime is not None:
        if start_datetime is None:
            start_datetime = pd.Timestamp.min
        if end_datetime is None:
            end_datetime = pd.Timestamp.max
        if not end_datetime < start_datetime:
            # start in time period, end in time period, or are ongoing in time period
            fire_info = fire_info[((fire_info["start"] >= start_datetime) &
                                   (fire_info["start"] <= end_datetime)) |
                                  ((fire_info["end"] >= start_datetime) &
                                   (fire_info["end"] <= end_datetime)) |
                                  ((fire_info["start"] < start_datetime) &
                                   (fire_info["end"] > end_datetime))]

    # keep fires with equal or larger spread as minimum spread
    if min_spread > 0:
        fire_info = fire_info[fire_info["spread"] >= min_spread]

    return fire_info


def write_fire_info(fire_info, networks=None, main_columns=False, index_column=False, start_datetime=None,
                    end_datetime=None, min_spread=0):
    """
    Writes fire info to file.

    :param fire_info: fire info DataFrame
    :param networks: network code or network code list to select info from networks
    :param main_columns: include only network code, latitude, longitude, spread, start date and end date columns if
                         True, otherwise include all columns
    :param index_column: include index column if True
    :param start_datetime: start datetime for info as Timestamp object or None to use the minimum pandas Timestamp
                           (ignored if start_datetime and end_datetime parameters are None)
    :param end_datetime: end datetime for info as Timestamp object or None to use the maximum pandas Timestamp
                         (ignored if start_datetime and end_datetime parameters are None)
    :param min_spread: minimum fire spread in hectares for info
    """
    # keep selected networks
    if networks is not None:
        if get_data_label(networks) == "str":
            networks = [networks]
        fire_info = fire_info[fire_info["network"].isin(set(networks))]

    # keep selected columns
    if main_columns:
        select_columns = {"network", "lat", "lon", "spread", "start", "end"}
        fire_info = fire_info[fire_info.columns[fire_info.columns.isin(select_columns)]]

    # keep fires in time range
    if start_datetime is not None or end_datetime is not None:
        if start_datetime is None:
            start_datetime = pd.Timestamp.min
        if end_datetime is None:
            end_datetime = pd.Timestamp.max
        if not end_datetime < start_datetime:
            # start in time period, end in time period, or are ongoing in time period
            fire_info = fire_info[((fire_info["start"] >= start_datetime) &
                                   (fire_info["start"] <= end_datetime)) |
                                  ((fire_info["end"] >= start_datetime) &
                                   (fire_info["end"] <= end_datetime)) |
                                  ((fire_info["start"] < start_datetime) &
                                   (fire_info["end"] > end_datetime))]

    # keep fires with equal or larger spread as minimum spread
    if min_spread > 0:
        fire_info = fire_info[fire_info["spread"] >= min_spread]

    write_data_to_file(fire_info, "fires.csv", join_path("data", root=True), index=index_column)


def read_station_data(station, start_datetime, end_datetime, column="", network="", category="fetched", subcategory="",
                      main_columns=False, set_index_column="valid"):
    """
    Reads station data from files.

    Station data files are saved under ./data/[network]/[station]/[category]/ for fetched and modeled data and
    ./data/[network]/[station]/[category]/[subcategory]/ for preprocessed data.

    The names of the files are [station]_[start_datetime]_[end_datetime].csv and
    [station]_[start_datetime]_[end_datetime]_[column].csv for column specific data.

    Brackets above reference function parameter values, start_datetime and end_datetime values are in YYYYmmddHHMM
    (year, month, day, hour, minute) format.

    :param station: station ID ("ALL" for 24h data from all stations)
    :param start_datetime: start datetime for data as Timestamp object
    :param end_datetime: end datetime for data as Timestamp object
    :param column: column name to get a column specific data file (column specific data must exist for the combination
                   of category and subcategory parameters)
    :param network: station network code ("ALL" for 24h data from all stations), if "" get network code from station
                    info file using station parameter
    :param category: select kind of data being read -
                     fetched data if "fetched",
                     preprocessed data if "preprocessed",
                     modeled data if "modeled"
    :param subcategory: used with category parameter to select king of data being read -
                        if category parameter is "preprocessed" -
                        clean data if "clean" or "",
                        clean data scaled with MinMaxScaler if "scaled_minmax",
                        clean data scaled with StandardScaler if "scaled_standard"
                        (split column data if column parameter is not "")
                        if category parameter is "modeled" -
                        split clean data clustered using K-Means clustering with DTW if "kmeans_timeseries"
    :param main_columns: include only timestamp, temperature, dew point temperature, relative humidity, wind speed,
                         sky level 1 coverage code, sky level 1 altitude, and apparent temperature columns if True and
                         column parameter is not a column name, otherwise include all columns (ignored if column
                         parameter is not "")
    :param set_index_column: column name to set as index, if "valid" parse timestamp column as DatetimeIndex, if None
                             don't set index from file columns
    :return: station data DataFrame or None if reading failed
    """
    if category == "fetched":
        subcategory = ""
        column = ""
    elif category == "preprocessed":
        if subcategory == "":
            subcategory = "clean"
    elif category == "modeled":
        if subcategory == "":
            subcategory = "kmeans_timeseries"
            column = ""
    else:
        return None

    # get network
    if network == "":
        station_info = read_station_info(stations=[station])
        if station_info is not None:
            if station_info.empty:
                return
        network = station_info["network"].iloc[0]

    if set_index_column == "valid":
        parse_dates = True
    else:
        parse_dates = False

    if column == "":
        # column specific data
        file = join_file(station, start_datetime, end_datetime, sep="_", ext=".csv", convert=True)
    else:
        file = join_file(station, start_datetime, end_datetime, column, sep="_", ext=".csv", convert=True)
    directory = join_path("data", network, station, category, subcategory, root=True)

    if category == "fetched":
        data_types = get_station_data_dtypes()
        station_data = read_data_from_file(file, directory, index_col=set_index_column, parse_dates=parse_dates,
                                           dtype=data_types)
    elif category == "preprocessed":
        station_data = read_data_from_file(file, directory, index_col=set_index_column, parse_dates=parse_dates,
                                           dtype="float64")
    elif category == "modeled":
        station_data = read_data_from_file(file, directory, index_col=set_index_column, parse_dates=parse_dates)
    else:
        return None

    # keep selected columns
    if main_columns and column == "" and station_data is not None:
        select_columns = {"tmpf", "dwpf", "relh", "sknt", "skyc1", "skyl1", "feel"}
        station_data = station_data[station_data.columns[station_data.columns.isin(select_columns)]]

    return station_data


def write_station_data(station_data, station, start_datetime, end_datetime, column="", network="", category="fetched",
                       subcategory="", main_columns=False, index_column=True):
    """
    Writes station data from files.

    Station data files are saved under ./data/[network]/[station]/[category]/ for fetched and modeled data and
    ./data/[network]/[station]/[category]/[subcategory]/ for preprocessed data.

    The names of the files are [station]_[start_datetime]_[end_datetime].csv and
    [station]_[start_datetime]_[end_datetime]_[column].csv for column specific data.

    Brackets above reference function parameter values, start_datetime and end_datetime values are in YYYYmmddHHMM
    (year, month, day, hour, minute) format.

    :param station_data: station data DataFrame
    :param station: station ID ("ALL" for 24h data from all stations)
    :param start_datetime: start datetime for data as Timestamp object
    :param end_datetime: end datetime for data as Timestamp object
    :param column: column name to indicate writing a column specific data file
    :param network: station network code ("ALL" for 24h data from all stations), if "" get network code from station
                    info file using station parameter
    :param category: select kind of data being written -
                     fetched data if "fetched",
                     preprocessed data if "preprocessed",
                     modeled data if "modeled"
    :param subcategory: used with category parameter to select kind of data being written -
                        if category parameter is "preprocessed" -
                        clean data if "clean" or "",
                        clean data scaled with MinMaxScaler if "scaled_minmax",
                        clean data scaled with StandardScaler if "scaled_standard"
                        (split column data if column parameter is not "")
                        if category parameter is "modeled" -
                        split clean data clustered using K-Means clustering with DTW if "kmeans_timeseries"
    :param main_columns: include only timestamp, temperature, dew point temperature, relative humidity, wind speed,
                         sky level 1 coverage code, sky level 1 altitude, and apparent temperature columns if True and
                         column parameter is not a column name, otherwise include all columns (ignored if column
                         parameter is not "")
    :param index_column: include index column if True
    """
    if category == "fetched":
        subcategory = ""
        column = ""
    elif category == "preprocessed":
        if subcategory == "":
            subcategory = "clean"
    elif category == "modeled":
        if subcategory == "":
            subcategory = "kmeans_timeseries"
            column = ""
    else:
        return

    if network == "":
        station_info = read_station_info(stations=[station])
        if station_info is not None:
            if station_info.empty:
                return
        network = station_info["network"].iloc[0]

    if column == "":
        # column specific data
        file = join_file(station, start_datetime, end_datetime, sep="_", ext=".csv", convert=True)

        # keep selected columns
        if main_columns:
            select_columns = {"tmpf", "dwpf", "relh", "sknt", "skyc1", "skyl1", "feel"}
            station_data = station_data[station_data.columns[station_data.columns.isin(select_columns)]]
    else:
        file = join_file(station, start_datetime, end_datetime, column, sep="_", ext=".csv", convert=True)
    directory = join_path("data", network, station, category, subcategory, root=True)

    write_data_to_file(station_data, file, directory, index=index_column)


def get_station_data_dtypes(column=""):
    """
    Returns data types for station data columns.

    :param column: column name or "" for all columns
    :return: data type for selected column if column parameter is a column or None if column is invalid, otherwise
             columns and data types dictionary
    """
    column_dtypes_dict = {
        "station": "object",
        "valid": "object",
        "tmpf": "float64",
        "dwpf": "float64",
        "relh": "float64",
        "drct": "float64",
        "sknt": "float64",
        "p01i": "float64",
        "alti": "float64",
        "mslp": "float64",
        "vsby": "float64",
        "gust": "float64",
        "skyc1": "object", "skyc2": "object", "skyc3": "object", "skyc4": "object",
        "skyl1": "float64", "skyl2": "float64", "skyl3": "float64", "skyl4": "float64",
        "wxcodes": "object",
        "ice_accretion_1hr": "float64", "ice_accretion_3hr": "float64", "ice_accretion_6hr": "float64",
        "peak_wind_gust": "float64", "peak_wind_drct": "float64", "peak_wind_time": "object",
        "feel": "float64",
        "metar": "object",
        "snowdepth": "float64"
    }

    if column != "":
        return column_dtypes_dict.get(column)
    else:
        return column_dtypes_dict


def get_station_data_descriptions(column="", preprocessed=False, units=False):
    """
    Returns descriptions for station data columns.

    :param column: column name or "" for all columns
    :param preprocessed: descriptions for preprocessed station data instead of fetched data if True
    :param units: include units of measurement in descriptions if True
    :return: description for selected column if column parameter is a column or "" if column is invalid, otherwise
             columns and descriptions dictionary
    """
    if not preprocessed:
        if units:
            column_descriptions_dict = {
                "station": "Station Identifier",
                "valid": "Observation Timestamp",
                "tmpf": "Air Temperature [°F]",
                "dwpf": "Dew Point Temperature [°F]",
                "relh": "Relative Humidity [%]",
                "drct": "Wind Direction from True North [°]",
                "sknt": "Wind Speed [kt]",
                "p01i": "Precipitation over 1h [in]",
                "alti": "Pressure Altimeter [inHg]",
                "mslp": "Sea Level Pressure [mbar]",
                "vsby": "Visibility [mi]",
                "gust": "Wind Gust [kt]",
                "skyc1": "Sky Level 1 Coverage Code",
                "skyc2": "Sky Level 2 Coverage Code",
                "skyc3": "Sky Level 3 Coverage Code",
                "skyc4": "Sky Level 4 Coverage Code",
                "skyl1": "Sky Level 1 Altitude [ft]",
                "skyl2": "Sky Level 2 Altitude [ft]",
                "skyl3": "Sky Level 3 Altitude [ft]",
                "skyl4": "Sky Level 4 Altitude [ft]",
                "wxcodes": "Present Weather Codes",
                "feel": "Apparent Temperature [°F]",
                "ice_accretion_1hr": "Ice Accretion over 1h [in]",
                "ice_accretion_3hr": "Ice Accretion over 3h [in]",
                "ice_accretion_6hr": "Ice Accretion over 6h [in]",
                "peak_wind_gust": "Peak Wind Gust [kt]",
                "peak_wind_drct": "Peak Wind Gust Direction [°]",
                "peak_wind_time": "Peak Wind Gust Time",
                "metar": "Unprocessed Reported Observation in METAR Format",
                "snowdepth": "Snow Depth (Missing)"
            }
        else:
            column_descriptions_dict = {
                "station": "Station Identifier",
                "valid": "Observation Timestamp",
                "tmpf": "Air Temperature",
                "dwpf": "Dew Point Temperature",
                "relh": "Relative Humidity",
                "drct": "Wind Direction from True North",
                "sknt": "Wind Speed",
                "p01i": "Precipitation over 1h",
                "alti": "Pressure Altimeter",
                "mslp": "Sea Level Pressure",
                "vsby": "Visibility",
                "gust": "Wind Gust",
                "skyc1": "Sky Level 1 Coverage Code",
                "skyc2": "Sky Level 2 Coverage Code",
                "skyc3": "Sky Level 3 Coverage Code",
                "skyc4": "Sky Level 4 Coverage Code",
                "skyl1": "Sky Level 1 Altitude",
                "skyl2": "Sky Level 2 Altitude",
                "skyl3": "Sky Level 3 Altitude",
                "skyl4": "Sky Level 4 Altitude",
                "wxcodes": "Present Weather Codes",
                "feel": "Apparent Temperature",
                "ice_accretion_1hr": "Ice Accretion over 1h",
                "ice_accretion_3hr": "Ice Accretion over 3h",
                "ice_accretion_6hr": "Ice Accretion over 6h",
                "peak_wind_gust": "Peak Wind Gust",
                "peak_wind_drct": "Peak Wind Gust Direction",
                "peak_wind_time": "Peak Wind Gust Time",
                "metar": "Unprocessed Reported Observation in METAR Format",
                "snowdepth": "Snow Depth (Missing)"
            }
    else:
        if units:
            column_descriptions_dict = {
                "valid": "Observation Timestamp",
                "tmpf": "Air Temperature [°F]",
                "dwpf": "Dew Point Temperature [°F]",
                "relh": "Relative Humidity [%]",
                "sknt": "Wind Speed [kt]",
                "p01i": "Precipitation over 1h [in]",
                "vsby": "Visibility [mi]",
                "gust": "Wind Gust [kt]",
                "skyc1": "Sky Level 1 Coverage Code (CLR/FEW/SCT/BKN/OVC/VV ordinal encoding)",
                "skyc2": "Sky Level 2 Coverage Code (CLR/FEW/SCT/BKN/OVC/VV ordinal encoding)",
                "skyc3": "Sky Level 3 Coverage Code (CLR/FEW/SCT/BKN/OVC/VV ordinal encoding)",
                "skyl1": "Sky Level 1 Altitude [ft]",
                "skyl2": "Sky Level 2 Altitude [ft]",
                "skyl3": "Sky Level 3 Altitude [ft]",
                "feel": "Apparent Temperature [°F]",
                "drct_sin": "Wind Direction from True North (sine cyclical encoding)",
                "drct_cos": "Wind Direction from True North (cosine cyclical encoding)",
            }
        else:
            column_descriptions_dict = {
                "valid": "Observation Timestamp",
                "tmpf": "Air Temperature",
                "dwpf": "Dew Point Temperature",
                "relh": "Relative Humidity",
                "sknt": "Wind Speed",
                "p01i": "Precipitation over 1h",
                "vsby": "Visibility",
                "gust": "Wind Gust",
                "skyc1": "Sky Level 1 Coverage Code (CLR/FEW/SCT/BKN/OVC/VV ordinal encoding)",
                "skyc2": "Sky Level 2 Coverage Code (CLR/FEW/SCT/BKN/OVC/VV ordinal encoding)",
                "skyc3": "Sky Level 3 Coverage Code (CLR/FEW/SCT/BKN/OVC/VV ordinal encoding)",
                "skyl1": "Sky Level 1 Altitude",
                "skyl2": "Sky Level 2 Altitude",
                "skyl3": "Sky Level 3 Altitude",
                "feel": "Apparent Temperature",
                "drct_sin": "Wind Direction from True North (sine cyclical encoding)",
                "drct_cos": "Wind Direction from True North (cosine cyclical encoding)",
            }

    if column != "":
        return column_descriptions_dict.get(column, "")
    else:
        return column_descriptions_dict
