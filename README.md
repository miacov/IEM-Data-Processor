# IEM Data Processor
Tool to scrape and process station data from the Iowa Environmental Mesonet (IEM).

## Table of Contents
* [Usage](#usage)
  * [Examples](#examples)


* [Selecting Networks, Stations, and Time Range](#selecting-networks-stations-and-time-range)
  * [Configuring Config Files](#configuring-config-files)
  * [Selecting with Config Files](#selecting-with-config-files)

  
* [Fetching Data](#fetching-data)
  * [Fetching Network and Station Info](#fetching-network-and-station-info)
  * [Fetching Station Data](#fetching-station-data)


* [Preprocessing Data](#preprocessing-data)
  * [Cleaning Fetched Station Data](#cleaning-fetched-station-data)
  * [Scaling Clean Station Data](#scaling-clean-station-data)
  * [Splitting Clean Station Data](#splitting-clean-station-data)
  * [Combining Preprocessing Steps](#combining-preprocessing-steps)

  
* [Modeling Data](#modeling-data)
  * [Clustering Split Clean Station Data](#clustering-split-clean-station-data)


* [Mapping Data](#mapping-data)


* [Plotting Data](#plotting-data)

 
* [Inputting and Outputting Data](#inputting-and-outputting-data)
  * [Inputting and Outputting Network, Station, and Fire Info](#inputting-and-outputting-network-station-and-fire-info)
  * [Inputting and Outputting Station Data](#inputting-and-outputting-station-data)


* [Data and Info Column Descriptions](#data-and-info-column-descriptions)
  * [Station Data Descriptions](#station-data-descriptions)
  * [Station Info Descriptions](#station-info-descriptions)
  * [Network Info Descriptions](#network-info-descriptions)
  * [Fire Info Descriptions](#fire-info-descriptions)


* [Required Libraries](#required-libraries)



## Usage
* [Fetching](#fetching-data) functions can be found in `fetching.py`.


* [Preprocessing](#preprocessing-data) functions (including transformations) can be found in `preprocessing.py`.


* [Modeling](#modeling-data) functions can be found in `modeling.py`.


* [Mapping](#mapping-data) functions can be found in `mapping.py`.


* [Plotting](#plotting-data) functions can be found in `plotting.py`.


* Helper functions (e.g., [selections](#selecting-networks-stations-and-time-range),
  [I/O](#inputting-and-outputting-data)) can be found in `utils.py`.


### Examples
* Usage examples can be found under `./examples/`:
  * Examples of [fetching](#fetching-data), [preprocessing](#preprocessing-data), and [modeling](#modeling-data) data
    are included in `example_fetching_preprocessing_modeling.py` and in the Jupyter notebook
    `example_fetching_preprocessing_modeling.ipynb` with visual examples of a demo.
  * Examples of plotting and mapping data for visualization are included in `example_mapping_plotting.py` and in
    the Jupyter notebook `example_mapping_plotting.ipynb` with visual examples of a demo.
  * Examples of inputting and outputting data using storage directories are included in `example_io.py` and in the
    Jupyter notebook `example_io.ipynb` with visual examples of a demo.


* The functions used in the examples make curated use of several other functions included with the tool. Refer to
  function documentation under `./doc/` for additional info on other functions for advanced usage.



## Selecting Networks, Stations, and Time Range
* In order to use tool functions, a selections must be made of networks, stations, and time range of data.


* Selections can be made using the config files saved under `./config/` to
[get lists of selections](#selecting-with-config-files) or by creating the lists without config files.


* If network codes and station IDs used for selections are not known, they can be
[fetched](#fetching-network-and-station-info) first from the IEM for reference.

### Configuring Config Files
* The networks config (`networks.txt`) must have a network code at each line:
```
NETWORK_1
NETWORK_3
...
NETWORK_N
```


* The stations config (`stations.txt`) must have a station ID at each line:
```
STATION_1
STATION_2
...
STATION_N
``` 


* The datetimes config (`datetimes.txt`) must have a start datetime and an end datetime indicating the time range for
  the data. The format dd/mm/YYYY HH:MM:SS (day/month/year hour/minute/second), where time is optional (treated as
  00:00:00 if omitted) must be used for the datetimes:
```
dd/mm/YYYY HH:MM:SS
dd/mm/YYYY HH:MM:SS
```

### Selecting with Config Files
* Config options can be selected using the parameters `network_config`, `stations_config`, and `datetimes_config`:
```python
# select networks with config file
network_list = utils.read_config_options(networks_config=True)

# select stations with config file
station_list = utils.read_config_options(stations_config=True)

# select time range for data with config file
start_datetime, end_datetime = utils.read_config_options(datetimes_config=True)
```


* Config files can also be used to [input or output info files](#inputting-and-outputting-network-station-and-fire-info)
  after [fetching](#fetching-data) them.



## Fetching Data
* [Network info](#network-info-descriptions), [station info](#station-info-descriptions), and
  [station data](#station-data-descriptions) can be fetched from the IEM.


### Fetching Network and Station Info
* Network info can be fetched for all networks 


* Station info can be fetched for selected networks using the `networks` parameter:
```python
# fetch network info of all networks
fetch.fetch_and_output_network_info()

# fetch station info of selected networks
fetching.fetch_and_output_station_info(networks=network_list)
```


* Fetched network and station info files are saved under `./data/` with the names `networks.csv` and `stations.csv`.


### Fetching Station Data
* Station data from the IEM can be fetched for selected stations for a time range or for all available stations from all 
  available networks (default) up to a time range of 24 hours, using parameters:
  * `start_datetime`: start of data time range
  * `end_datetime`: end of data time range (up to 24 hours after start_datetime parameter if fetching data for all
    stations)
  * `stations`: station ID, station ID list, station info DataFrame, if None then all stations


* Reports for station data can be all (default) or limited using the `report` parameter:
  * `report="frequent"`: MADIS HFRADAR (5-minute ASOS)
  * `report="routine"`: routine (once hourly)
  * `report="special"`: specials
  * `report="combined"`: routine and specials


* Timezone for presentation of observations can be changed from UTC (default) using the `timezone` parameter.


* Trace reports can either be empty (default) or represented with `0.0001` using `trace=True`. 
```python
# fetch station data for selected time range and stations
fetching.fetch_and_output_station_data(start_datetime, end_datetime, station_list)

# fetch station data of all stations up to a period of 24 hours
# reports are limited to routine and specials, timestamp timezone is UTC, trace reports are left in data
fetching.fetch_and_output_station_data(start_datetime=pd.to_datetime("2022-01-01"),
                                       end_datetime=pd.to_datetime("2022-01-02"),
                                       report="combined", timezone="Etc/UTC", trace=True)
```


* Fetched station data files are saved under `./data/NETWORK/STATION/fetched/` with the names `STATION_START_END.csv`:
  * `NETWORK` is the network code (replaced with `ALL` if fetching 24h data from all stations)
  * `STATION` is the station ID (replaced with `ALL` if fetching 24h data from all stations)
  * `START` is the start of the time range and `END` is the end of the time range with format YYYYmmddHHMM (year, month,
    day, hour, minute)


* In order to save fetched station data files of selected stations,
  [station info of the selected stations must be fetched using their networks](#fetching-network-and-station-info)
  first.



## Preprocessing Data
* [Fetched station data](#fetching-station-data) can be preprocessed to be ready for [modeling](#modeling-data).


* Preprocessing consists of [cleaning](#cleaning-fetched-station-data), [scaling](#scaling-clean-station-data), and
  [splitting](#splitting-clean-station-data) station data.


* Data can be preprocessed for selected stations for a time range (same used during [fetching](#fetching-station-data))
  using parameters:
  * `start_datetime`: start of data time range
  * `end_datetime`: end of data time range
  * `stations`: station ID, station ID list, station info DataFrame


### Cleaning Fetched Station Data
* The first part of preprocessing is cleaning fetched station data:
  * Unimportant columns are dropped
  * String columns are encoded with ordinal encoding
  * Cyclical columns are encoded with cyclical encoding
  * Data is resampled to hourly intervals.


* A limited selection of important [columns](#station-data-descriptions) (default) or all important columns can be
  cleaned using the `main_columns` parameter:
  * `main_columns=True`(default): clean only timestamp, temperature, dew point temperature, relative humidity, wind
    speed, sky level 1 coverage code, sky level 1 altitude, and apparent temperature columns
  * `main_columns=False`: additionally clean wind direction, precipitation, visibility, wind gust, sky level
    2-3 coverage codes, and sky level 2-3 altitude columns
```python
# clean fetched station data for selected time range and stations
# clean all important columns, not only the main important columns
preprocessing.clean_and_output_station_data(start_datetime, end_datetime, station_list,
                                            main_columns=False)
```

* Clean station data files are saved under `./data/NETWORK/STATION/preprocessed/clean/` with the names
  `STATION_START_END.csv`:
  * `NETWORK` is the network code
  * `STATION` is the station ID
  * `START` is the start of the time range and `END` is the end of the time range with format YYYYmmddHHMM (year, month,
    day, hour, minute)


* In order to save clean station data files of selected stations,
  [station info of the selected stations must be fetched using their networks](#fetching-network-and-station-info)
  first.


### Scaling Clean Station Data
* After fetched station data is [cleaned](#cleaning-fetched-station-data), it can be scaled with selected scalers. 


* A limited selection of important [columns](#station-data-descriptions) (default) or all important columns can be
  scaled using the `main_columns` parameter:
  * `main_columns=True`(default): scale only timestamp, temperature, dew point temperature, relative humidity, wind
    speed, sky level 1 coverage code, sky level 1 altitude, and apparent temperature columns
  * `main_columns=False`: additionally scale wind direction, precipitation, visibility, wind gust, sky level
    2-3 coverage codes, and sky level 2-3 altitude columns (if they exist in clean station data file)


* Scalers used for scaling can be selected with the `scaler` parameter. Input can either be a scaler name as a string,
  or a list of scaler names:
  * `scaler="minmax"`(default): use scikit-learn MinMaxScaler
  * `scaler="standard"`: use scikit-learn StandardScaler
  * `scaler=["minmax", "standard"]`: use scikit-learn MinMaxScaler and StandardScaler


* Scalers selected with the `scaler` parameter can either be loaded from files or be created from scratch using the
  `new_scaler` parameter:
  * `new_scaler=True`: create a new default scaler, fit it to clean data of each station, and output it to file
  * `new_scaler=False`(default): use existing scaler from file (if a scaler file doesn't exist, a new scaler is created
    as when `new_scaler=True`)
```python
# scale clean station data for selected time range and stations
# scale only the main important columns
# scale clean data using saved MinMaxScaler and StandardScaler (new if scaler files don't exist)
preprocessing.scale_and_output_station_data(start_datetime, end_datetime, station_list,
                                            main_columns=True,
                                            scaler=["minmax", "standard"], new_scaler=False)

# scale clean station data for selected time range and stations
# scale all important columns, not only the main important columns
# scale clean data using saved MinMaxScaler (new if scaler file doesn't exist)
preprocessing.scale_and_output_station_data(start_datetime, end_datetime, station_list,
                                            main_columns=False,
                                            scaler="minmax", new_scaler=False)
```


* Scaled clean station data files are saved under `./data/NETWORK/STATION/preprocessed/scaled_SCALER/` with the names
  `STATION_START_END.csv`, and scalers are saved under `./models/NETWORK/STATION/scalers/` with the names
  `STATION_SCALER` if `main_columns=False` and `STATION_SCALER_main` if `main_columns=True`:
  * `NETWORK` is the network code
  * `STATION` is the station ID
  * `SCALER`  is the scaler name 
  * `START` is the start of the time range and `END` is the end of the time range with format YYYYmmddHHMM (year, month,
    day, hour, minute)


* In order to save scaled clean station data files of selected stations,
  [station info of the selected stations must be fetched using their networks](#fetching-network-and-station-info)
  first.


### Splitting Clean Station Data
* [Clean](#cleaning-fetched-station-data) and [scaled](#scaling-clean-station-data) station data columns can be split
  into daily time series.


* Each data column is separately split into daily rows with 24 hourly columns.


* A limited selection of important [columns](#station-data-descriptions) (default) or all important columns can be split
  using the `main_columns` parameter:
  * `main_columns=True`(default): split only timestamp, temperature, dew point temperature, relative humidity, wind
    speed, sky level 1 coverage code, sky level 1 altitude, and apparent temperature columns
  * `main_columns=False`: additionally split wind direction, precipitation, visibility, wind gust, sky level
    2-3 coverage codes, and sky level 2-3 altitude columns (if they exist in clean station data file or scaled clean
    data file)


* Files with data columns to be split can be selected with the `split_columns` parameter:
  * `split_columns="clean"`(default): split clean station data columns (if file exists)
  * `split_columns="scaled"`: split scaled clean station data columns (if file exists), with scaled clean data files
    selected with the `scaler` parameter
  * `split_columns="all"`: split both clean and scaled clean station data columns


* Scaled clean data files can be selected with the `scaler` parameter to be split instead of being ignored (default). 
  Input can either be a scaler name as a string, or a list of scaler names:
  * `scaler="minmax"`: use clean data scaled with scikit-learn MinMaxScaler
  * `scaler="standard"`: use clean data scaled with scikit-learn StandardScaler
  * `scaler=["minmax", "standard"]`: use clean data scaled with scikit-learn MinMaxScaler and StandardScaler
```python
# split clean station data for selected time range and stations
# split only the main important columns
# split only clean data columns
preprocessing.split_and_output_station_data(start_datetime, end_datetime, station_list,
                                            main_columns=True,
                                            split_columns="clean")

# split clean station data for selected time range and stations
# split only the main important columns
# split both clean and scaled clean data columns
# for scaled data split only clean data scaled with MinMaxScaler
preprocessing.split_and_output_station_data(start_datetime, end_datetime, station_list,
                                            main_columns=True,
                                            split_columns="all", scaler="minmax")
```  


* Split clean station data files are saved under `./data/NETWORK/STATION/preprocessed/clean/`, and split scaled clean
  data files are saved under `./data/NETWORK/STATION/preprocessed/scaled_SCALER/` (one file for each split column), with
  the names `STATION_START_END_COLUMN.csv`:
  * `NETWORK` is the network code
  * `STATION` is the station ID
  * `SCALER`  is the scaler name 
  * `START` is the start of the time range and `END` is the end of the time range with format YYYYmmddHHMM (year, month,
    day, hour, minute)
  * `COLUMN` is the split column name


* In order to save split clean and clean scaled station data files of selected stations,
  [station info of the selected stations must be fetched using their networks](#fetching-network-and-station-info)
  first.


### Combining Preprocessing Steps
* Preprocessing steps of [cleaning](#cleaning-fetched-station-data), [scaling](#scaling-clean-station-data), and
  [splitting](#splitting-clean-station-data) station data can be performed with just one function.


* The function takes parameters previously described separately at each step:
  * `main_columns` selects [columns](#station-data-descriptions) for all selected steps (`main_columns=True` as default)
  * `split_columns` selects which clean data will be split (`split_columns=""` as default to skip splitting by default)
  * `scaler` selects which scaler will be used for scaling (`scaler=None` as default to skip scaling by default) 
  * `new_scaler` selects whether a new scaler will be created (`new_scaler=False` as default)
```python
# preprocess fetched station data for selected time range and stations
# preprocess only main important columns
# split only clean data columns
# scale clean data using saved MinMaxScaler (new if scaler file doesn't exist)
preprocessing.preprocess_and_output_station_data(start_datetime, end_datetime, station_list,
                                                 main_columns=True,
                                                 split_columns="clean",
                                                 scaler="minmax", new_scaler=False)
```



## Modeling Data
* [Preprocessed station data](#preprocessing-data) can be used to build models.


* Data can be modeled for selected stations for a time range (same used during [fetching](#fetching-station-data) and
  [preprocessing](#preprocessing-data)) using parameters:
  * `start_datetime`: start of data time range
  * `end_datetime`: end of data time range
  * `stations`: station ID, station ID list, station info DataFrame


### Clustering Split Clean Station Data
* [Split clean station data](#splitting-clean-station-data) files (clean data columns split into daily rows with 24
  hourly columns) can each be clustered into a number of clusters.


* Clustering each split file allows for categorization of each station data column for each day.


* Columns clustered are temperature, dew point temperature, relative humidity, wind speed, sky level 1 coverage code, 
  sky level 1 altitude, and apparent temperature.


* The number of clusters can be changed from 5 (default) using the `clusters` parameter.


* Clustering models be loaded from files or be created from scratch using the `new_model` parameter:
  * `new_model=True`: create new default tslearn TimeSeriesKMeans models (one for each split clean data file), fit each
    to a supported split clean data column, and output them to files
  * `new_model=False`(default): use existing models from files (if a model file doesn't exist, a new model is created
    as when `new_model=True`)
```python
# cluster split clean station data for selected time range and stations
# create 15 clusters for each split column
# cluster data using saved TimeSeriesKMeans models (new if model files don't exist)
modeling.cluster_and_output_station_data(start_datetime, end_datetime, station_list,
                                         clusters=15, new_model=False)
```


* Clustered station data files are saved under `./data/NETWORK/STATION/modeled/kmeans_timeseries/` with the names
  `STATION_START_END.csv`, and models are saved under `./models/NETWORK/STATION/kmeans_timeseries/` with the names
  `STATION_COLUMN`:
  * `NETWORK` is the network code
  * `STATION` is the station ID
  * `START` is the start of the time range and `END` is the end of the time range with format YYYYmmddHHMM (year, month,
    day, hour, minute)


* In order to save clustered station data files of selected stations,
  [station info of the selected stations must be fetched using their networks](#fetching-network-and-station-info)
  first.



## Mapping Data
* In order to visualize the locations of stations and fires in networks, info maps can be created using
  [station info](#station-info-descriptions) and [fire info](#fire-info-descriptions).


* Maps include station location markers, fire location circles with spread, and lines connecting each fire with the 
  station nearest to it.


* For fires specifically, another map can be created showing active fires for each year.


* A time range can be set to select only stations that start service, end service, or are active in it and only fires
  that start, end, or are ongoing in it using parameters:
  * `start_datetime`: start of info time range
  * `end_datetime`: end of info time range


* An info map and a yearly fire info map can be created for all networks combined (default) or a separate info map and
  yearly fire info map can be created for each network using the `networks` parameter.
  

* Maps can just be outputted as HTML files (default) or additionally screenshotted and outputted as image files using
  `screenshot=True`.
```python
# make info maps for selected time range and networks
# additionally output screenshots of maps
mapping.make_and_output_network_info_maps(start_datetime=start_datetime, end_datetime=end_datetime,
                                          networks=network_list,
                                          screenshot=True)

# make info maps for all available networks combined for selected time range
# additionally output screenshots of maps
mapping.make_and_output_network_info_maps(start_datetime=start_datetime, end_datetime=end_datetime,
                                          screenshot=True)


# make info maps for all available networks combined without a time range
# additionally output screenshots of maps
mapping.make_and_output_network_info_maps(screenshot=True)
```  


* Map files are saved under `./maps/NETWORK/` with the names `NETWORK_START_END.html` for info maps,
  `NETWORK_START_END.png` for info map screenshots, `NETWORK_START_END_fires.html` for yearly fire info maps, and
  `NETWORK_START_END_fires.png` for yearly fire info map screenshots (`_START` and/or `_END` missing if corresponding
  time range values not selected):
  * `NETWORK` is the network code (replaced with `ALL` creating maps for all networks combined)
  * `START` is the start of the time range and `END` is the end of the time range with format YYYYmmddHHMM (year, month,
    day, hour, minute)



## Plotting Data
* [Fetched station data](#fetching-station-data), preprocessed [clean](#cleaning-fetched-station-data) and
  [scaled clean station data](#scaling-clean-station-data), and
  [clustered split clean station data](#clustering-split-clean-station-data) can be plotted.


* [Fetched](#fetching-station-data) and [preprocessed station data](#preprocessing-data)
  [columns](#station-data-descriptions) are used to make data distribution and time plot figures:
  * Distribution plot figures are used to visualize data distribution and includes a count plot for missing values, a
    histogram plot, and a box plot (if column is numeric)
  * Time plot figures are used to visualize data trend and seasonality and include a line plot for data over time, a
    line plot and a scatter plot for monthly data, and grouped plots for monthly and quarterly data (if downsampled
    data doesn't have missing values and all months are available)


* [Modeled station data](#modeling-data) [columns](#station-data-descriptions) are used to make time series cluster plot
  figures:
  * Time series cluster plot figures are used to visualize time series clusters and include a plot for every cluster
    with cluster centers and clustered samples belonging to each center and a scatter plot with clustered samples on
    data over time


* Station data can be plotted using parameters to find the appropriate files:
  * `category`: select kind of data being plotted:
    * `category="fetched"`: fetched data
    * `category="preprocessed"`: preprocessed data
    * `category="modeled"`: modeled data
  * `subcategory`: used with category parameter to select the kind of data being plotted:
    * if `category="preprocessed"` selected:
      * `subcategory="clean"` or `subcategory="preprocessed"`: clean data
      * `subcategory="scaled_minmax"`: clean data scaled with MinMaxScaler
      * `subcategory="scaled_standard"`: clean data scaled with StandardScaler
    * if `category="modeled"` selected:
      * `subcategory="kmeans_timeseries"`: data clustered using K-Means clustering with DTW    


* Plotted [columns](#station-data-descriptions) for data distribution and time plot figures can be all (default) or only
  timestamp, temperature, dew point temperature, relative humidity, wind speed, sky level 1 coverage code, sky level 1
  altitude, and apparent temperature using `main_columns=True`.


* Plot figures can just be outputted (default) or additionally displayed using `display=True`.
```python
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
```


* Fetched data plot files are saved under `./plots/NETWORK/STATION/CATEGORY/`, and preprocessed and modeled data plot
  files are saved under `./plots/NETWORK/STATION/CATEGORY/SUBCATEGORY/`, with the names `STATION_START_END_COLUMN.png`:
  * `NETWORK` is the network code
  * `STATION` is the station ID  
  * `CATEGORY` is the plotted data category parameter and `SUBCATEGORY` is the plotted data subcategory parameter
  * `START` is the start of the time range and `END` is the end of the time range with format YYYYmmddHHMM (year, month,
    day, hour, minute)
  * `COLUMN` is the column name



## Inputting and Outputting Data
* This tool uses certain storage directories to save data to as previously described, in order to be able to access data
  needed for functions.


* Data can be inputted from and outputted to storage directories used by the tool manually.


### Inputting and Outputting Network, Station, and Fire Info
* Network info can be inputted or outputted for all networks (default) or selected networks using the `networks`
  parameter.


* Inputted or outputted [columns](#network-info-descriptions) can be all (default) or only the network code using
  `main_columns=True`.
```python
# input only network code of selected networks
network_info = utils.read_network_info(networks=network_list, main_columns=True)

# input all network info of every available network
network_info = utils.read_network_info()

# output all network info of every available network
utils.write_network_info(network_info)
```


* Station info can be inputted or outputted for all stations (default) or selected stations using the `stations`
  parameter or stations in selected networks using the `networks` parameter.


* A time range can be set to select only stations that start service, end service, or are active in it using parameters:
  * `start_datetime`: start of info time range
  * `end_datetime`: end of info time range


* Inputted or outputted [columns](#station-info-descriptions) can be all (default) or only the station ID and network
  code using `main_columns=True`.
```python
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
```


* Fire info can be inputted or outputted for all fires (default) or fires in selected networks using the `networks`
  parameter.


* A time range can be set to select only fires that start, end, or are ongoing in it using parameters:
  * `start_datetime`: start of info time range
  * `end_datetime`: end of info time range


* A minimum spread (in hectares) can be set to select only fires with equal or larger spread than it with the `min_spread`
  parameter.


* Inputted or outputted [columns](#fire-info-descriptions) can be all (default) or only the network code, latitude,
  longitude, spread, start datetime, and end datetime using `main_columns=True`.
```python
# input only network code, latitude, longitude, spread, and time period of fires in selected networks
fire_info = utils.read_fire_info(networks=network_list, main_columns=True)

# input all station info of active fires in selected time range with spread greater or equal to 5 hectares
fire_info = utils.read_fire_info(start_datetime=start_datetime, end_datetime=end_datetime,
                                 min_spread=5)

# input all fire info of every available fire
fire_info = utils.read_fire_info()

# output all fire info of every available fire
utils.write_fire_info(fire_info)
```


* When outputting info, the index column can be ignored (default) or be included using `index_column=True`.


### Inputting and Outputting Station Data
* Station data can be inputted and outputted using parameters to find the appropriate files:
  * `station`: station ID ("ALL" for 24h data from all stations)
  * `start_datetime`: start of data time range
  * `end_datetime`: end of data time range
  * `column`: column name to indicate inputting or outputting a column specific data file  
  * `network`: network code ("ALL" for 24h data from all stations), if not given then get network code from station info
    file using `station` parameter
  * `category`: select kind of data being inputted or outputted:
    * `category="fetched"`: fetched data
    * `category="preprocessed"`: preprocessed data
    * `category="modeled"`: modeled data
  * `subcategory`: used with category parameter to select the kind of data being inputted or outputted:
    * if `category="preprocessed"` selected:
      * `subcategory="clean"` or `subcategory="preprocessed"`: clean data
      * `subcategory="scaled_minmax"`: clean data scaled with MinMaxScaler (split column data if `column` parameter is
        not `""`)
      * `subcategory="scaled_standard"`: clean data scaled with StandardScaler (split column data if `column` parameter
        is not `""`)
    * if `category="modeled"` selected:
      * `subcategory="kmeans_timeseries"`: data clustered using K-Means clustering with DTW


* Inputted or outputted [columns](#station-data-descriptions) can be all (default) or only timestamp, temperature, dew
  point temperature, relative humidity, wind speed, sky level 1 coverage code, sky level 1 altitude, and apparent 
  temperature using `main_columns=True` (parameter is ignored when `column` parameter is not `""`).


* When inputting data, the index column can be changed from timestamp (default) using the `index_column` parameter.


* When outputting data, the index column can be included (default) or ignored using `index_column=False`.
```python
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
```



## Data and Info Column Descriptions
### Station Data Descriptions
| Column             | Description                                                                                  |
| ------------------:| -------------------------------------------------------------------------------------------- |
| station            |  Station Identifier (3 or 4 characters)                                                      |
| valid              |  Observation Timestamp                                                                       |
| tmpf               |  Air Temperature (typically at 2 meters) [°F (Fahrenheit)]                                   |
| dwpf               |  Dew Point Temperature (typically at 2 meters) [°F (Fahrenheit)]                             |
| relh               |  Relative Humidity [% (percentage)]                                                          |
| drct               |  Wind Direction from true north [° (degrees)]                                                |
| sknt               |  Wind Speed [kt (knots)]                                                                     |
| p01i               |  Precipitation over 1 Hour [in (inches)]                                                     |
| alti               |  Pressure Altimeter [inHg (inches Mercury)]                                                  |
| mslp               |  Sea Level Pressure [mbar (millibar)]                                                        |
| vsby               |  Visibility [mi (miles)]                                                                     |
| gust               |  Wind Gust [kt (knots)]                                                                      | 
| skyc1              |  Sky Level 1 Coverage Code (Clear/Few/Scattered/Broken/Overcast/Vertical Visibility)         |
| skyc2              |  Sky Level 2 Coverage Code (Clear/Few/Scattered/Broken/Overcast/Vertical Visibility)         |
| skyc3              |  Sky Level 3 Coverage Code (Clear/Few/Scattered/Broken/Overcast/Vertical Visibility)         |
| skyc4              |  Sky Level 4 Coverage Code (Clear/Few/Scattered/Broken/Overcast/Vertical Visibility)         |
| skyl1              |  Sky Level 1 Altitude [ft (feet)]                                                            |
| skyl2              |  Sky Level 2 Altitude [ft (feet)]                                                            |
| skyl3              |  Sky Level 3 Altitude [ft (feet)]                                                            |
| skyl4              |  Sky Level 4 Altitude [ft (feet)]                                                            |
| wxcodes            |  Present Weather Codes (space separated)                                                     |
| feel               |  Apparent Temperature [°F (Fahrenheit)]                                                      |
| ice_accretion_1hr  |  Ice Accretion over 1 Hour [in (inches)]                                                     |
| ice_accretion_3hr  |  Ice Accretion over 3 Hours [in (inches)]                                                    |
| ice_accretion_6hr  |  Ice Accretion over 6 Hours [in (inches)]                                                    |
| peak_wind_gust     |  Peak Wind Gust (from PK WND METAR remark) [kt (knots)]                                      |
| peak_wind_drct     |  Peak Wind Gust Direction (from PK WND METAR remark) [° (degrees)]                           |
| peak_wind_time     |  Peak Wind Gust Time (from PK WND METAR remark)                                              |
| metar              |  Unprocessed Reported Observation in METAR Format (info split into other variables, dropped) |
| snowdepth          |  Snow Depth (missing report, dropped)                                                        |

* More info:  
  * [ASOS User's Guide](https://www.weather.gov/media/asos/aum-toc.pdf)
  * [IEM ASOS-AWOS-METAR Data Download Page](http://www.mesonet.agron.iastate.edu/request/download.phtml)


### Station Info Descriptions
| Column    | Description                            |
| ---------:| -------------------------------------- |
| elevation | Elevation above sea water [m (meters)] |
| sname     | Station name                           |
| state     | State code                             |
| country   | Country code                           |
| tzname    | Timezone name                          |
| county    | County name                            |
| sid       | Station ID                             |
| network   | Network code                           |
| start     | Year archive starts                    |
| end       | Year archive ends                      |
| lat       | Latitude [° (degrees)]                 |
| lon       | Longitude [° (degrees)]                |


### Network Info Descriptions
| Column       | Description  |
| ------------:| ------------ |
| network      | Network code |
| network_name | Network name |


### Fire Info Descriptions
| Column    | Description             |
| ---------:| ----------------------- |
| network   | Network code            |
| country   | Country code            |
| state     | State code              |
| location  | Fire location info      |
| lat       | Latitude [° (degrees)]  |
| lon       | Longitude [° (degrees)] |
| spread    | Spread [ha (hectares)]  |
| start     | Datetime fire starts    |
| end       | Datetime fire ends      |
| cause     | Fire cause info         |



## Required Libraries
* Mandatory Libraries
  * pandas
  * NumPY
  * scikit-learn
  * tslearn
  * Matplotlib
  * seaborn
  * statsmodels
  * Folium
  * Beautiful Soup
  * joblib


* Optional Libraries:
  * IPython
  * Jupyter
  * h5py
  * html2image
  * utm
