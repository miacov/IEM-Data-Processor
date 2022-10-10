import mapping
import plotting
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


# ============
# MAPPING DATA
# ============

# make info maps for selected time range and networks
# additionally output screenshots of maps
mapping.make_and_output_network_info_maps(start_datetime=start_datetime, end_datetime=end_datetime,
                                          networks=network_list,
                                          screenshot=True)

# make info map for all available networks combined without a time range
# additionally output screenshots of maps
mapping.make_and_output_network_info_maps(screenshot=True)


# =============
# PLOTTING DATA
# =============
# make fetched station data distribution and time plot figures for selected time range and stations
plotting.make_and_output_station_data_plots(start_datetime, end_datetime, station_list,
                                            category="fetched")

# make preprocessed station data distribution and time plot figures for selected time range and stations
# plot clean data
plotting.make_and_output_station_data_plots(start_datetime, end_datetime, station_list,
                                            category="preprocessed", subcategory="clean")

# make preprocessed station data distribution and time plot figures for selected time range and stations
# plot clean data scaled with MinMaxScaler
# plot only main columns
plotting.make_and_output_station_data_plots(start_datetime, end_datetime, station_list,
                                            category="preprocessed", subcategory="scaled_minmax",
                                            main_columns=True)

# make modeled station data time series cluster plot figures for selected time range and stations
# plot split clean station data clustered using K-Means clustering with DTW
plotting.make_and_output_station_data_plots(start_datetime, end_datetime, station_list,
                                            category="modeled", subcategory="kmeans_timeseries")
