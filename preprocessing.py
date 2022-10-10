import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotting
import utils


def split_train_test_data(data, labels=None, **options):
    """
    Splits data into train set and test set, after separating feature and label columns.

    :param data: data DataFrame
    :param labels: output labels
    :param options: variable number of keyword arguments for splitting options, see
                    sklearn.model_selection.train_test_split documentation
    :return: train and test feature and label DataFrames (X train, X test, Y train, Y test) or train and test feature
             DataFrames (X train, X test) if labels parameter is None
    """
    if labels is not None:
        x_data = data[data.columns[~data.columns.isin(labels)]]
        y_data = data[data.columns[data.columns.isin(labels)]]

        return train_test_split(x_data, y_data, **options)
    else:
        return train_test_split(data, **options)


def fit_scaler(data, scaler="minmax", **options):
    """
    Fits scaler to data.

    :param data: data DataFrame
    :param scaler: sklearn MinMaxScaler or StandardScaler model or "minmax" to make new MinMaxScaler model or "standard"
                   to make new StandardScaler model
    :param options: variable number of keyword arguments for scaling options, see sklearn.preprocessing.MinMaxScaler or
                    sklearn.preprocessing.StandardScaler documentation
    :return: fitted scaler or None if making or fitting new scaler failed
    """
    # make scaler if no model given
    try:
        if scaler == "minmax":
            scaler = MinMaxScaler(**options)
        elif scaler == "standard":
            scaler = StandardScaler(**options)
    except TypeError:
        return None

    # fit data
    try:
        return scaler.fit(data)
    except (TypeError, AttributeError):
        return None


def scale_data(data, scaler="minmax", get_model=False, **options):
    """
    Scales data.

    :param data: data DataFrame
    :param scaler: sklearn MinMaxScaler or StandardScaler model or "minmax" to make new MinMaxScaler model or "standard"
                   to make new StandardScaler model
    :param get_model: return scaler with scaled data if True, otherwise return only scaled data
    :param options: variable number of keyword arguments for scaling options, see sklearn.preprocessing.MinMaxScaler or
                    sklearn.preprocessing.StandardScaler documentation
    :return: scaled data DataFrame (only columns fitted are scaled) or None if scaling data or fitting new scaler failed
             and additionally the new scaler (scaled data, scaler) or None if scaling data or fitting new scaler failed
             (None, None) if get_model parameter is True
    """
    if scaler in {"minmax", "standard"}:
        scaler = fit_scaler(data, scaler, **options)

    try:
        # handle column label mismatch between current columns and fitted scaler columns
        has_string_columns = hasattr(scaler, "feature_names_in_")
        if has_string_columns:
            fitted_columns = set(scaler.feature_names_in_)  # fitted columns
            columns = set(data.columns)  # current columns
            unexpected_columns = sorted(columns - fitted_columns)  # columns not fitted
            missing_columns = sorted(fitted_columns - columns)  # fitted columns missing
            # drop columns not fitted
            if unexpected_columns:
                data = data.drop(unexpected_columns, axis=1, errors="ignore")
            # add empty columns for missing columns (dropped after scaling, avoids transform error)
            if missing_columns:
                data = pd.concat([data, pd.DataFrame(columns=missing_columns, index=data.index)], axis=1)
            data = data[scaler.feature_names_in_]  # change column order to fitted column order

        # scale data
        scaled_data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

        # drop empty columns added for missing columns
        if has_string_columns:
            if missing_columns is not None:
                scaled_data = scaled_data.drop(missing_columns, axis=1)

        if get_model:
            return scaled_data, scaler
        else:
            return scaled_data
    except (TypeError, ValueError, AttributeError):
        if get_model:
            return None, None
        else:
            return None


def smooth_lowess_column(column, data=None, display=False, **options):
    """
    Smooths a column with LOWESS (Locally Weighted Scatterplot Smoothing)

    :param column: value Series or column name (requires data parameter to get value Series)
    :param data: data DataFrame (if column parameter is name)
    :param display: display plot figure of column values before and after smoothing if True
    :param options: variable number of keyword arguments for smoothing options, see
                    statsmodels.nonparametric.smoothers_lowess.lowess documentation
    :return: smooth value Series
    """
    column_type = utils.get_data_label(column)
    if column_type in {"str", "int"}:
        if data is not None:
            try:
                column = data[column]
            except KeyError:
                return None
        else:
            return None
    elif column_type != "series":
        return None

    try:
        column_name = column.name
        smooth_column_name = str(column_name) + "_smooth"
        column_data = column.reset_index(drop=False)  # save previous index

        smooth_column = lowess(exog=column_data.index, endog=column_data[column_name], **options)

        # return_sorted=True
        if len(smooth_column.shape) == 2:
            column_data = column_data.reindex(smooth_column[:, 0])
            column_data[smooth_column_name] = smooth_column[:, 1]
        # return_sorted=False
        else:
            column_data[smooth_column_name] = smooth_column
        column_data = column_data.set_index(column_data.columns[0])

        if display:
            option_frac = options.get("frac", 0.6666666666666666)
            option_it = options.get("it", 3)
            option_delta = options.get("delta", 0.0)
            description = "LOWESS (frac={frac}, it={it}, delta={delta})".format(frac=option_frac, it=option_it,
                                                                                delta=option_delta)
            plotting.make_lowess_plot(column_name, smooth_column_name, column_data, description=description,
                                  display=True, get=False)

        return column_data[smooth_column_name]
    except TypeError:
        return None


def split_column(split, column, data=None, index=None, smooth=False, scale=False, **options):
    """
    Splits a column into rows with selected amount of columns.

    :param split: amount of columns each row will have after splitting
    :param column: value Series or column name (requires data parameter to get value Series)
    :param data: data DataFrame (if column parameter is name)
    :param index: index for rows after splitting
    :param smooth: smooth rows using LOWESS after splitting if True
    :param scale: scale rows (after smoothing if smooth parameter is True) using sklearn MinMaxScaler or StandardScaler
                  if True
    :param options: variable number of keyword arguments for options -
                    if smoothing see smooth_lowess_column documentation (display and options parameters)
                    if scaling see scale_data documentation (scaler and options parameters)
    :return: split column DataFrame (smooth if smooth parameter is True, scaled if scale parameter is True, smooth and
             scaled if both parameters are True, with index if index parameter is not None) or None if any
             transformation failed
    """
    column_type = utils.get_data_label(column)
    if column_type in {"str", "int"}:
        if data is not None:
            try:
                column = data[column]
            except KeyError:
                return None
        else:
            return None
    elif column_type != "series":
        return None

    # pad column to fit reshape
    rows = int(np.ceil(column.size/split))
    pad_count = rows*split - column.size
    if pad_count != 0:
        column = pd.concat([column, pd.Series([np.nan for x in range(pad_count)])], ignore_index=True)
        column = column.ffill()

    # split column to rows
    try:
        split_data = pd.DataFrame(column.values.reshape(-1, split))
    except ValueError:
        return None

    # smooth rows
    if smooth:
        option_display = options.pop("display", False)
        option_frac = options.pop("frac", 0.6666666666666666)
        option_it = options.pop("it", 3)
        option_delta = options.pop("delta", 0.0)
        option_xvals = options.pop("xvals", None)
        option_is_sorted = options.pop("is_sorted", True)
        option_missing = options.pop("missing", "drop")
        option_return_sorted = options.pop("return_sorted", True)

        split_data = split_data.apply(smooth_lowess_column, data=None, display=option_display,
                                      frac=option_frac, it=option_it, delta=option_delta, xvals=option_xvals,
                                      is_sorted=option_is_sorted, missing=option_missing,
                                      return_sorted=option_return_sorted,
                                      axis=1)

        if split_data is None:
            return None

    # scale rows
    if scale:
        option_scaler = options.pop("scaler", "minmax")
        option_copy = options.pop("copy", True)
        if option_scaler == "minmax":
            option_feature_range = options.pop("feature_range", (0, 1))
            option_clip = options.pop("clip", False)

            split_data = scale_data(split_data.T, scaler=option_scaler,
                                    copy=option_copy, feature_range=option_feature_range, clip=option_clip)
        elif option_scaler == "standard":
            option_with_mean = options.pop("with_mean", True)
            option_with_std = options.pop("with_std", True)

            split_data = scale_data(split_data.T, scaler=option_scaler,
                                    copy=option_copy, with_mean=option_with_mean, with_std=option_with_std)

        if split_data is None:
            return None
        else:
            split_data = split_data.T

    # add index
    if index is not None:
        split_data = split_data.set_index(index)

    return split_data


def encode_cyclical_column(max_value, column, data=None):
    """
    Encodes a column with cyclical encoding (sine, cosine).

    :param max_value: column maximum value
    :param column: value Series or column name (requires data parameter to get value Series)
    :param data: data DataFrame (if column parameter is name)
    :return: encoded sine and cosine DataFrame if column parameter is a Series, data parameter DataFrame with
             encoded sine and cosine column columns added and previous column dropped if data parameter is not None,
             or None if encoding failed
    """
    column_type = utils.get_data_label(column)
    if column_type in {"str", "int"}:
        if data is not None:
            try:
                column = data[column]
            except KeyError:
                return None
        else:
            return None
    elif column_type != "series":
        return None
    column_name = column.name

    column_norm = 2 * np.pi * column/max_value  # normalize for 0-2pi cycle
    column_sin = pd.Series(np.sin(column_norm), name=str(column_name) + "_sin")  # encode sin
    column_cos = pd.Series(np.cos(column_norm), name=str(column_name) + "_cos")  # encode cos

    cyclical_data = pd.concat([column_sin, column_cos], axis=1)

    if data is not None:
        data = data.drop(column_name, axis=1, errors="ignore")
        cyclical_data = pd.concat([data, cyclical_data], axis=1)  # add to existing dataframe

    return cyclical_data


def clean_station_data(station_data, main_columns=True, frequency="H"):
    """
    Cleans station data.

    Drops unimportant columns, encodes string columns with ordinal encoding, encodes cyclical columns with cyclical
    encoding, and resamples data.

    :param station_data: station data DataFrame
    :param main_columns: include only timestamp, temperature, dew point temperature, relative humidity, wind speed,
                         sky level 1 coverage code, sky level 1 altitude, and apparent temperature columns if True
                         otherwise additionally include wind direction, precipitation, visibility, wind gust, sky level
                         2-3 coverage codes, and sky level 2-3 altitude columns
    :param frequency: frequency of samples for resampling
    :return: preprocessed station data DataFrame
    """
    if not isinstance(station_data.index, pd.DatetimeIndex):
        if station_data.index.name == "valid":
            station_data.index = pd.to_datetime(station_data.index)
        else:
            station_data.index = pd.to_datetime(station_data["valid"])
            station_data = station_data.drop("valid", axis=1)

    # drop columns
    drop = ["station", "lat", "lon", "elevation", "alti", "mslp", "skyc4", "skyl4", "wxcodes",
            "ice_accretion_1hr", "ice_accretion_3hr", "ice_accretion_6hr",
            "peak_wind_gust", "peak_wind_drct", "peak_wind_time", "metar", "snowdepth"]
    if main_columns:
        drop.extend(["drct", "p01i", "vsby", "gust", "skyc2", "skyc3", "skyl2", "skyl3"])
    station_data = (
        station_data.drop(drop, axis=1, errors="ignore")
                    .dropna(axis=1, how="all")
    )
    columns = set(station_data.columns)  # columns after drop

    # convert columns before resampling
    # sky codes to int with ordinal encoding
    sky_ordinal_encoder = OrdinalEncoder(categories=[["CLR", "FEW", "SCT", "BKN", "OVC", "VV"]],
                                         handle_unknown="use_encoded_value", unknown_value=np.nan)
    if "skyc1" in columns:
        station_data["skyc1"] = station_data["skyc1"].str.replace(" ", "")
        station_data[["skyc1"]] = sky_ordinal_encoder.fit_transform(station_data[["skyc1"]])

    if not main_columns:
        # gust and precipitation as True or False (1,0) indications
        if "gust" in columns:
            station_data["gust"] = (station_data["gust"] > 0).astype(int)
        if "p01i" in columns:
            station_data["p01i"] = (station_data["p01i"] > 0).astype(int)

        # sky codes to int with ordinal encoding
        if "skyc2" in columns:
            station_data["skyc2"] = station_data["skyc2"].str.replace(" ", "")
            station_data["skyc2"] = sky_ordinal_encoder.fit_transform(station_data[["skyc2"]])
        if "skyc3" in columns:
            station_data["skyc3"] = station_data["skyc3"].str.replace(" ", "")
            station_data["skyc3"] = sky_ordinal_encoder.fit_transform(station_data[["skyc3"]])

    # resample
    aggregate_dict = {
        "tmpf": "mean",
        "dwpf": "mean",
        "relh": "mean",
        "drct": "median",
        "sknt": "mean",
        "p01i": "max",
        "vsby": "mean",
        "gust": "max",
        "skyc1": "median", "skyc2": "median", "skyc3": "median", "skyc4": "median",
        "skyl1": "median", "skyl2": "median", "skyl3": "median", "skyl4": "median",
        "feel": "mean"
    }
    aggregate_dict = dict((key, value) for key, value in aggregate_dict.items() if key in columns)
    station_data = station_data.groupby(pd.Grouper(freq=frequency)).agg(aggregate_dict)

    # fill missing values
    columns_interpolate = {"tmpf", "dwpf", "relh", "sknt", "vsby", "feel"}
    columns_ordinal = {"skyc1", "skyc2", "skyc3"}
    columns_truefalse = {"p01i", "gust"}
    for column in columns:
        if column in columns_interpolate:
            station_data[column] = station_data[column].interpolate(method="time", limit_direction="both")
        elif column in columns_truefalse:
            station_data[column] = station_data[column].fillna(0)
        else:
            # replace decimal values for ordinal columns
            if column in columns_ordinal:
                station_data[column] = station_data[column].apply(lambda x: x if x in range(6) else np.nan)
            station_data[column] = station_data[column].ffill().bfill()

    # wind direction from degrees to sine and cosine
    if not main_columns and "drct" in columns:
        station_data = encode_cyclical_column(360, "drct", station_data)

    return station_data


def clean_and_output_station_data(start_datetime, end_datetime, stations, main_columns=True):
    """
    Cleans fetched station data and outputs it to files.

    Clean station data is in hourly intervals, with empty and unimportant columns dropped, string columns encoded with
    ordinal encoding, and cyclical columns encoded with cyclical encoding.

    :param start_datetime: start datetime for data as Timestamp object
    :param end_datetime: end datetime for data as Timestamp object
    :param stations: station ID, station ID list, or station info DataFrame
    :param main_columns: include only timestamp, temperature, dew point temperature, relative humidity, wind speed,
                         sky level 1 coverage code, sky level 1 altitude, and apparent temperature columns if True
                         otherwise additionally include wind direction, precipitation, visibility, wind gust, sky level
                         2-3 coverage codes, and sky level 2-3 altitude columns
    """
    if utils.get_data_label(stations) == "list":
        stations = utils.read_station_info(stations=stations, main_columns=True)
    if stations is None:
        return

    for sid, network in zip(stations["sid"], stations["network"]):
        # input fetched data
        station_data = utils.read_station_data(sid, start_datetime, end_datetime,
                                               network=network, category="fetched", main_columns=main_columns)
        # no data for station
        if station_data is None:
            continue

        # clean data
        station_data = clean_station_data(station_data, main_columns=main_columns, frequency="H")

        # output clean data
        if station_data is not None:
            utils.write_station_data(station_data, sid, start_datetime, end_datetime,
                                     network=network, category="preprocessed", subcategory="clean")


def scale_and_output_station_data(start_datetime, end_datetime, stations, main_columns=True, scaler=None,
                                  new_scaler=False):
    """
    Scales clean station data and outputs it to files.

    :param start_datetime: start datetime for data as Timestamp object
    :param end_datetime: end datetime for data as Timestamp object
    :param stations: station ID, station ID list, or station info DataFrame
    :param main_columns: include only timestamp, temperature, dew point temperature, relative humidity, wind speed,
                         sky level 1 coverage code, sky level 1 altitude, and apparent temperature columns if True
                         otherwise additionally include wind direction, precipitation, visibility, wind gust, sky level
                         2-3 coverage codes, and sky level 2-3 altitude columns (if they exist in read clean data file)
    :param scaler: scaler or list of scalers to scale clean data, sklearn MinMaxScaler model if "minmax", StandardScaler
                   model if "standard" ("minmax" selected if no scaler is given)
    :param new_scaler: create a new default scaler for every scaler selected by the scaler parameter, fit it to clean
                       data, and output it to file if True, otherwise use existing scaler from file (if the file exists)
                       to scale data (ignored if no valid scaler parameter is given)
    """
    if utils.get_data_label(stations) == "list":
        stations = utils.read_station_info(stations=stations, main_columns=True)
    if stations is None:
        return

    for sid, network in zip(stations["sid"], stations["network"]):
        # input clean data
        station_data = utils.read_station_data(sid, start_datetime, end_datetime,
                                               network=network, category="preprocessed", subcategory="clean",
                                               main_columns=main_columns)
        # no data for station
        if station_data is None:
            continue

        # scale clean data
        if scaler is None:
            scaler = ["minmax"]
        if utils.get_data_label(scaler) == "str":
            scaler = list(scaler.split())
        for scaler_name in scaler:
            if scaler_name not in {"minmax", "standard"}:
                continue
            # handle scaler
            scaler_object = None
            if main_columns:
                scaler_file = utils.join_file(sid, scaler_name, "main", sep="_", ext=".joblib")
            else:
                scaler_file = utils.join_file(sid, scaler_name, sep="_", ext=".joblib")
            scaler_directory = utils.join_path("models", network, sid, "scalers", root=True)
            if not new_scaler:
                scaler_object = utils.read_data_from_file(scaler_file, scaler_directory)
            # read scaler doesn't exist or making new scaler
            if scaler_object is None:
                scaler_object = fit_scaler(station_data, scaler=scaler_name)
                if scaler_object is not None:
                    utils.write_data_to_file(scaler_object, scaler_file, scaler_directory)

            scaled_station_data = scale_data(station_data, scaler_object)

            # output scaled preprocessed data
            if scaled_station_data is not None:
                utils.write_station_data(scaled_station_data, sid, start_datetime, end_datetime,
                                         network=network, category="preprocessed", subcategory="scaled_" + scaler_name)


def split_and_output_station_data(start_datetime, end_datetime, stations, main_columns=True, split_columns="clean",
                                  scaler=None):
    """
    Splits clean station data columns into daily time series and outputs them to files.

    Each data column is separately split into daily rows with 24 hourly columns.

    :param start_datetime: start datetime for data as Timestamp object
    :param end_datetime: end datetime for data as Timestamp object
    :param stations: station ID, station ID list, or station info DataFrame
    :param main_columns: include only timestamp, temperature, dew point temperature, relative humidity, wind speed,
                         sky level 1 coverage code, sky level 1 altitude, and apparent temperature columns if True
                         otherwise additionally include wind direction, precipitation, visibility, wind gust, sky level
                         2-3 coverage codes, and sky level 2-3 altitude columns (if they exist in read clean data file)
    :param split_columns: data to split each data column of in daily time series, clean data if "clean", scaled clean
                          data if "scaled" (scaled data selected with scaler parameter), both clean and scaled clean
                          data if "all", otherwise no splitting
    :param scaler: scaler or list of scalers that were used to scale clean data to get appropriate scaled data files for
                   splitting, sklearn MinMaxScaler model if "minmax", StandardScaler model if "standard" (ignored if
                   split_columns parameter is not "scaled" or "all")
    """
    if utils.get_data_label(stations) == "list":
        stations = utils.read_station_info(stations=stations, main_columns=True)
    if stations is None:
        return

    for sid, network in zip(stations["sid"], stations["network"]):
        # split clean data
        if split_columns in {"clean", "all"}:
            # input clean station data
            station_data = utils.read_station_data(sid, start_datetime, end_datetime,
                                                   network=network, category="preprocessed", subcategory="clean",
                                                   main_columns=main_columns)

            if station_data is not None:
                dates_unique = station_data.asfreq("d").index
                for column in station_data.columns:
                    # split column into timeseries rows with 24 columns (1 for each hour of the day)
                    split_column_data = split_column(24, column, station_data, index=dates_unique,
                                                     smooth=True, frac=0.4)

                    # output split preprocessed column data
                    if split_column_data is not None:
                        utils.write_station_data(split_column_data, sid, start_datetime, end_datetime, column,
                                                 network=network, category="preprocessed", subcategory="clean")

        # split scaled data
        if split_columns in {"scaled", "all"} and scaler is not None:
            if utils.get_data_label(scaler) == "str":
                scaler = list(scaler.split())
            for scaler_name in scaler:
                if scaler_name not in {"minmax", "standard"}:
                    continue
                # input scaled clean station data
                scaled_station_data = utils.read_station_data(sid, start_datetime, end_datetime,
                                                              network=network, category="preprocessed",
                                                              subcategory="scaled_" + scaler_name,
                                                              main_columns=main_columns)

                if scaled_station_data is not None:
                    dates_unique = scaled_station_data.asfreq("d").index
                    for column in scaled_station_data.columns:
                        # split column into timeseries rows with 24 columns (1 for each hour of the day)
                        split_column_data = split_column(24, column, scaled_station_data, index=dates_unique,
                                                         smooth=True, frac=0.4)

                        # output split preprocessed column data
                        if split_column_data is not None:
                            utils.write_station_data(split_column_data, sid, start_datetime, end_datetime, column,
                                                     network=network,
                                                     category="preprocessed", subcategory="scaled_" + scaler_name)


def preprocess_and_output_station_data(start_datetime, end_datetime, stations, main_columns=True, split_columns="",
                                       scaler=None, new_scaler=False):
    """
    Preprocesses fetched station data and outputs it to files.

    Preprocessing consists of cleaning, scaling, and splitting data. Clean station data is in hourly intervals, with
    empty and unimportant columns dropped, string columns encoded with ordinal encoding, and cyclical columns encoded
    with cyclical encoding. After cleaning, data can also be scaled and columns can be are separately split into daily
    time series if the appropriate parameters are selected.

    :param start_datetime: start datetime for data as Timestamp object
    :param end_datetime: end datetime for data as Timestamp object
    :param stations: station ID, station ID list, or station info DataFrame
    :param main_columns: include only timestamp, temperature, dew point temperature, relative humidity, wind speed,
                         sky level 1 coverage code, sky level 1 altitude, and apparent temperature columns if True
                         otherwise additionally include wind direction, precipitation, visibility, wind gust, sky level
                         2-3 coverage codes, and sky level 2-3 altitude columns
    :param split_columns: data to split each data column of in daily time series, clean data if "clean", scaled clean
                          data if "scaled" (scaled data selected with scaler parameter), both clean and scaled clean
                          data if "all", otherwise no splitting
    :param scaler: scaler or list of scalers to scale clean data, sklearn MinMaxScaler model if "minmax", StandardScaler
                   model if "standard"
    :param new_scaler: create a new default scaler for every scaler selected by the scaler parameter, fit it to clean
                       data, and output it to file if True, otherwise use existing scaler from file (if the file exists)
                       to scale data (ignored if no valid scaler parameter is given)
    """
    if utils.get_data_label(stations) == "list":
        stations = utils.read_station_info(stations=stations, main_columns=True)
    if stations is None:
        return

    for sid, network in zip(stations["sid"], stations["network"]):
        # input fetched data
        station_data = utils.read_station_data(sid, start_datetime, end_datetime,
                                               network=network, category="fetched", main_columns=main_columns)
        # no data for station
        if station_data is None:
            continue

        # clean data
        station_data = clean_station_data(station_data, main_columns=main_columns, frequency="H")

        # output clean data
        if station_data is not None:
            utils.write_station_data(station_data, sid, start_datetime, end_datetime,
                                     network=network, category="preprocessed", subcategory="clean")

            # split clean columns to daily time series
            if split_columns in {"clean", "all"}:
                dates_unique = station_data.asfreq("d").index
                for column in station_data.columns:
                    # split column into timeseries rows with 24 columns (1 for each hour of the day)
                    split_column_data = split_column(24, column, station_data, index=dates_unique,
                                                     smooth=True, frac=0.4)

                    # output split preprocessed column data
                    if split_column_data is not None:
                        utils.write_station_data(split_column_data, sid, start_datetime, end_datetime, column,
                                                 network=network, category="preprocessed", subcategory="clean")

        # scale clean data
        if scaler is None:
            continue
        if utils.get_data_label(scaler) == "str":
            scaler = list(scaler.split())
        for scaler_name in scaler:
            if scaler_name not in {"minmax", "standard"}:
                continue
            # handle scaler
            scaler_object = None
            if main_columns:
                scaler_file = utils.join_file(sid, scaler_name, "main", sep="_", ext=".joblib")
            else:
                scaler_file = utils.join_file(sid, scaler_name, sep="_", ext=".joblib")
            scaler_directory = utils.join_path("models", network, sid, "scalers", root=True)
            if not new_scaler:
                scaler_object = utils.read_data_from_file(scaler_file, scaler_directory)
            # read scaler doesn't exist or making new scaler
            if scaler_object is None:
                scaler_object = fit_scaler(station_data, scaler=scaler_name)
                if scaler_object is not None:
                    utils.write_data_to_file(scaler_object, scaler_file, scaler_directory)

            scaled_station_data = scale_data(station_data, scaler_object)

            # output scaled preprocessed data
            if scaled_station_data is not None:
                utils.write_station_data(scaled_station_data, sid, start_datetime, end_datetime,
                                         network=network, category="preprocessed", subcategory="scaled_" + scaler_name)

                # split scaled columns to daily time series
                if split_columns in {"scaled", "all"}:
                    dates_unique = scaled_station_data.asfreq("d").index
                    for column in scaled_station_data.columns:
                        # split column into timeseries rows with 24 columns (1 for each hour of the day)
                        split_column_data = split_column(24, column, scaled_station_data, index=dates_unique,
                                                         smooth=True, frac=0.4)

                        # output split preprocessed column data
                        if split_column_data is not None:
                            utils.write_station_data(split_column_data, sid, start_datetime, end_datetime, column,
                                                     network=network,
                                                     category="preprocessed", subcategory="scaled_" + scaler_name)
