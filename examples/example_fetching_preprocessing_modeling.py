import pandas as pd
import fetching
import preprocessing
import modeling
import utils

# ============================================
# SELECTING NETWORKS, STATIONS, AND TIME RANGE
# ============================================

# select networks with config file
network_list = utils.read_config_options(networks_config=True)

# select stations with config file
station_list = utils.read_config_options(stations_config=True)

# select time range for data with config file
start_datetime, end_datetime = utils.read_config_options(datetimes_config=True)


# =============
# FETCHING DATA
# =============

# fetch network info of all networks
fetching.fetch_and_output_network_info()

# fetch station info of selected networks
fetching.fetch_and_output_station_info(networks=network_list)

# fetch station data for selected time range and stations
fetching.fetch_and_output_station_data(start_datetime, end_datetime, station_list)

"""
# fetch station data of all stations up to a period of 24 hours
# reports are limited to routine and specials, timestamp timezone is UTC, trace reports are left in data
fetching.fetch_and_output_station_data(start_datetime=pd.to_datetime("2022-01-01"),
                                       end_datetime=pd.to_datetime("2022-01-02"),
                                       report="combined", timezone="Etc/UTC", trace=True)
"""


# ==================
# PREPROCESSING DATA
# ==================

# clean fetched station data for selected time range and stations
# clean all important columns, not only the main important columns
preprocessing.clean_and_output_station_data(start_datetime, end_datetime, station_list,
                                            main_columns=False)

# scale clean station data for selected time range and stations
# scale only the main important columns
# scale clean data using saved MinMaxScaler and StandardScaler (new if scaler files don't exist)
preprocessing.scale_and_output_station_data(start_datetime, end_datetime, station_list,
                                            main_columns=True,
                                            scaler=["minmax", "standard"], new_scaler=False)

"""
# scale clean station data for selected time range and stations
# scale all important columns, not only the main important columns
# scale clean data using saved MinMaxScaler (new if scaler file doesn't exist)
preprocessing.scale_and_output_station_data(start_datetime, end_datetime, station_list,
                                            main_columns=False,
                                            scaler="minmax", new_scaler=False)
"""


# split clean station data for selected time range and stations
# split only the main important columns
# split only clean data columns
preprocessing.split_and_output_station_data(start_datetime, end_datetime, station_list,
                                            main_columns=True,
                                            split_columns="clean")

"""
# split clean station data for selected time range and stations
# split only the main important columns
# split both clean and scaled clean data columns
# for scaled data split only clean data scaled with MinMaxScaler
preprocessing.split_and_output_station_data(start_datetime, end_datetime, station_list,
                                            main_columns=True,
                                            split_columns="all", scaler="minmax")
"""

"""

# preprocess fetched station data for selected time range and stations
# preprocess only main important columns
# split only clean data columns
# scale clean data using saved MinMaxScaler (new if scaler file doesn't exist)
preprocessing.preprocess_and_output_station_data(start_datetime, end_datetime, station_list,
                                                 main_columns=True,
                                                 split_columns="clean",
                                                 scaler="minmax", new_scaler=False)
"""


# =============
# MODELING DATA
# =============

# cluster split clean station data for selected time range and stations
# create 15 clusters for each split column
# cluster data using saved TimeSeriesKMeans models (new if model files don't exist)
modeling.cluster_and_output_station_data(start_datetime, end_datetime, station_list,
                                         clusters=15, new_model=True)
