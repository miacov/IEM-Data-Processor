import pandas as pd
import utils

# Note: To run this file without errors, run example_fetching_preprocessing_modeling.py first
# to generate the required data files

# ============================================
# SELECTING NETWORKS, STATIONS, AND TIME RANGE
# ============================================

# select networks with config file
network_list = utils.read_config_options(networks_config=True)

# select stations with config file
station_list = utils.read_config_options(stations_config=True)

# select time range for data with config file
start_datetime, end_datetime = utils.read_config_options(datetimes_config=True)


# =============================
# INPUTTING AND OUTPUTTING DATA
# =============================

# network info
# input only network code of selected networks
network_info = utils.read_network_info(networks=network_list, main_columns=True)

# input all network info of every available network
network_info = utils.read_network_info()

# output all network info of every available network
utils.write_network_info(network_info)


# station info
# input only station ID and network code of selected stations
station_info = utils.read_station_info(stations=station_list, main_columns=True)

# input only station ID and network code of stations in selected networks
station_info = utils.read_station_info(networks=network_list, main_columns=True)

# input all station info of every active station in time range
station_info = utils.read_station_info(start_datetime=start_datetime, end_datetime=end_datetime)

# input all station info of every available station
station_info = utils.read_station_info()

# output all station info of every available station
utils.write_station_info(station_info)


# fire info
# input only network code, latitude, longitude, spread, and time period of fires in selected networks
fire_info = utils.read_fire_info(networks=network_list, main_columns=True)

# input all station info of active fires in selected time range with spread greater or equal to 5 hectares
fire_info = utils.read_fire_info(start_datetime=start_datetime, end_datetime=end_datetime,
                                 min_spread=5)

# input all fire info of every available fire
fire_info = utils.read_fire_info()

# output all fire info of every available fire
utils.write_fire_info(fire_info)


# station data
# fetched
# input fetched station data for station LGAV in network GR__ASOS for selected time range
# select timestamp column as index
station_data = utils.read_station_data(station="LGAV",
                                       start_datetime=start_datetime, end_datetime=end_datetime,
                                       network="GR__ASOS",
                                       category="fetched",
                                       set_index_column="valid")

# output fetched station data for station LGAV in network GR__ASOS for selected time range
# include index column
utils.write_station_data(station_data,
                         station="LGAV",
                         start_datetime=start_datetime, end_datetime=end_datetime,
                         network="GR__ASOS",
                         category="fetched",
                         index_column=True)

# input fetched station data from all stations up to a period of 24 hours
# input only main columns
# select timestamp column as index
station_data = utils.read_station_data(station="ALL",
                                       start_datetime=pd.to_datetime("2022-01-01"),
                                       end_datetime=pd.to_datetime("2022-01-02"),
                                       network="ALL",
                                       category="fetched",
                                       main_columns=True,
                                       set_index_column="valid")

# preprocessed
# input preprocessed station data scaled with MinMaxScaler
# for station LGAV in network GR__ASOS for selected time range
# input only main columns
# select timestamp column as index
station_data = utils.read_station_data(station="LGAV",
                                       start_datetime=start_datetime, end_datetime=end_datetime,
                                       network="GR__ASOS",
                                       category="preprocessed", subcategory="scaled_minmax",
                                       main_columns=True,
                                       set_index_column="valid")

# output preprocessed station data scaled with MinMaxScaler
# for station LGAV in network GR__ASOS for selected time range
# include index column
utils.write_station_data(station_data,
                         station="LGAV",
                         start_datetime=start_datetime, end_datetime=end_datetime,
                         network="GR__ASOS",
                         category="preprocessed", subcategory="scaled_minmax",
                         index_column=True)

# input preprocessed split clean station data of temperature column
# for station LGAV for selected time range
# infer station network code (no network given)
# select timestamp column as index
station_data = utils.read_station_data(station="LGAV",
                                       start_datetime=start_datetime, end_datetime=end_datetime,
                                       column="tmpf",
                                       category="preprocessed", subcategory="clean",
                                       set_index_column="valid")

# output preprocessed split clean station data of temperature column
# for station LGAV for selected time range
# infer station network code (no network given)
# include index column
utils.write_station_data(station_data,
                         station="LGAV",
                         start_datetime=start_datetime, end_datetime=end_datetime,
                         column="tmpf",
                         category="preprocessed", subcategory="clean",
                         index_column=True)

# modeled
# input clustered split clean station data for station LGAV in network GR__ASOS for selected time range
# select timestamp column as index
station_data = utils.read_station_data(station="LGAV",
                                       start_datetime=start_datetime, end_datetime=end_datetime,
                                       network="GR__ASOS",
                                       category="modeled", subcategory="kmeans_timeseries",
                                       set_index_column="valid")

# output clustered split clean station data for station LGAV in network GR__ASOS for selected time range
# infer station network code (no network given)
# include index column
utils.write_station_data(station_data,
                         station="LGAV",
                         start_datetime=start_datetime, end_datetime=end_datetime,
                         category="modeled", subcategory="kmeans_timeseries",
                         index_column=True)
