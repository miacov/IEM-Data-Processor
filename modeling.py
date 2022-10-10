import pandas as pd
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)  # h5py not installed warning
    from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
import utils


def fit_kmeans(data, model="kmeans", get_centers=False, **options):
    """
    Fits K-Means clustering model to data.

    :param data: data DataFrame with column features if model parameter is sklearn KMeans, with time series as rows if
                 model parameter is tslearn TimeSeriesKMeans
    :param model: sklearn KMeans or tslearn TimeSeriesKMeans model or "kmeans" to make new KMeans model or
                  "kmeans_timeseries" to make new TimeSeriesKMeans model
    :param get_centers: return cluster center DataFrame with clustered data if True, otherwise return only clustered
                        data
    :param options: variable number of keyword arguments for clustering options, see sklearn.cluster.KMeans or
                    tslearn.clustering.TimeSeriesKMeans documentation
    :return: fitted clustering model or None if making or fitting new model failed, and additionally the cluster centers
             DataFrame (model, clusters) or None if making or fitting new model failed (None, None) if get_centers
             parameter is True
    """
    try:
        if model == "kmeans":
            model = KMeans(**options)
        elif model == "kmeans_timeseries":
            model = TimeSeriesKMeans(**options)
    except TypeError:
        return None

    try:
        model = model.fit(data)
        if get_centers:
            return model, pd.DataFrame(np.squeeze(model.cluster_centers_))
        else:
            return model
    except (TypeError, AttributeError):
        if get_centers:
            return None, None
        else:
            return None


def cluster_kmeans_data(data, model="kmeans", get_model=False, **options):
    """
    Clusters data.

    :param data: data DataFrame with column features if model parameter is sklearn KMeans, with time series as rows if
                 model parameter is tslearn TimeSeriesKMeans
    :param model: sklearn KMeans or tslearn TimeSeriesKMeans model or "kmeans" to make new KMeans model or
                  "kmeans_timeseries" to make new TimeSeriesKMeans model
    :param get_model: return clustering model with clustered data if True, otherwise return only clustered data
    :param options: variable number of keyword arguments for clustering options, see sklearn.cluster.KMeans or
                    tslearn.clustering.TimeSeriesKMeans documentation
    :return: clustered data DataFrame or None if clustering data or fitting new model failed, and additionally the new
             model (clustered data, model) or None if clustering data or fitting new model failed (None, None) if
             get_model parameter is True
    """
    if model in {"kmeans", "kmeans_timeseries"}:
        model = fit_kmeans(data, model, **options)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)  # 2D data warning
            if get_model:
                return pd.Series(model.predict(data), index=data.index), model
            else:
                return pd.Series(model.predict(data), index=data.index)
    except (TypeError, ValueError, AttributeError):
        if get_model:
            return None, None
        else:
            return None


def cluster_and_output_station_data(start_datetime, end_datetime, stations, clusters=5, new_model=False):
    """
    Clusters split clean station data and outputs the clusters to files.

    Clustered data includes each column clustered into clusters using K-Means clustering with DTW (Dynamic Time
    Warping).

    Columns clustered are temperature, dew point temperature, relative humidity, wind speed, sky level 1 coverage code,
    sky level 1 altitude, and apparent temperature.

    :param start_datetime: start datetime for data as Timestamp object
    :param end_datetime: end datetime for data as Timestamp object
    :param stations: station ID, station ID list, or station info DataFrame
    :param clusters: number of clusters
    :param new_model: create new default tslearn TimeSeriesKMeans models (one for each split clean data file), fit each
                      to a supported split clean data column, and output them to files if True, otherwise use existing
                      models from files if the files exist to cluster columns
    """
    if utils.get_data_label(stations) == "list":
        stations = utils.read_station_info(stations=stations, main_columns=True)
    if stations is None:
        return

    for sid, network in zip(stations["sid"], stations["network"]):
        # cluster every column
        clustered_data = None
        for column in ["tmpf", "dwpf", "relh", "sknt", "skyc1", "skyl1", "feel"]:
            # input split preprocessed data for column
            split_column_data = utils.read_station_data(sid, start_datetime, end_datetime, column,
                                                        network=network,
                                                        category="preprocessed", subcategory="clean")
            # no data for column
            if split_column_data is None:
                continue

            # handle time series clustering model
            model_object = None
            model_file = utils.join_file(sid, column, sep="_", ext=".json")
            model_directory = utils.join_path("models", network, sid, "kmeans_timeseries", root=True)
            if not new_model:
                model_object = utils.read_data_from_file(model_file, model_directory, reading="tslearn",
                                                         type="json", mclass=TimeSeriesKMeans)
            # read model doesn't exist or making new model
            if model_object is None:
                model_object = fit_kmeans(split_column_data, model="kmeans_timeseries",
                                          n_clusters=clusters, metric="dtw")
                # output new model
                if model_object is not None:
                    utils.write_data_to_file(model_object, model_file, model_directory, type="json")

            clustered_column_data = cluster_kmeans_data(split_column_data, model_object)

            if clustered_column_data is not None:
                clustered_column_data = clustered_column_data.rename(column)
                # add clustered column to clustered columns df
                if clustered_data is None:
                    clustered_data = clustered_column_data
                else:
                    clustered_data = pd.concat([clustered_data, clustered_column_data], axis=1)

        # output clustered data
        if clustered_data is not None:
            utils.write_station_data(clustered_data, sid, start_datetime, end_datetime,
                                     network=network, category="modeled", subcategory="kmeans_timeseries")
