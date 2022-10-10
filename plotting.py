import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.cm as cm
import seaborn as sns
from statsmodels.graphics.tsaplots import month_plot, quarter_plot
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)  # h5py not installed warning
    from tslearn.clustering import TimeSeriesKMeans
import utils


def make_plot_from_image(image, dpi=100.0, display=True, get=False):
    """
    Makes plot figure from plot figure image.

    :param image: image array
    :param dpi: image DPI used when saving the image before reading it
    :param display: display plot figure if True (disable plot window to avoid errors when plot limit is reached)
    :param get: return plot figure if True, otherwise close created plot figure and return None
    :return:
    """
    # select figure options according to image size
    height, width, depth = image.shape
    fig = plt.figure(figsize=(width/dpi, height/dpi), constrained_layout=True)

    # image plot
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    ax.axis("off")

    if display:
        fig.show()

    if get:
        return fig
    else:
        plt.close(fig)
        return None


def make_timeseries_cluster_plot(column, model, data=None, split_data=None, index=None, description="", display=True,
                                 get=False, style="seaborn-darkgrid"):
    """
    Makes plot figure to visualize time series clusters.

    Figure includes a plot for every cluster with cluster centers and clustered samples belonging to each center
    and a scatter plot with clustered samples on data over time (clustered samples and scatter plot only plotted if
    split column data is given).

    :param column: cluster value Series or column name (requires data parameter to get value Series)
    :param model: fitted clustering model
    :param data: data DataFrame with column of column parameter included (used if column parameter is name)
    :param split_data: split column DataFrame with time series rows, if None plot only cluster centers
    :param index: index for rows of stacked split_data parameter DataFrame (original data before splitting), if None use
                  default index after stacking (used if split_data parameter is not None)
    :param description: description to add to plot figure title after column name, if "" add no description
    :param display: display plot figure if True (disable plot window to avoid errors when plot limit is reached)
    :param get: return plot figure if True, otherwise close created plot figure and return None
    :param style: style name for plot figure
    :return: plot figure if get parameter is True or None if get parameter is False or required column data is not given
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

    # handle clusters
    cluster_count = model.n_clusters
    cluster_colors = cm.rainbow(np.linspace(0, 1, cluster_count))

    # select figure options and handle data
    if split_data is not None:
        subplots_num = cluster_count + 1

        # stack split data
        stacked_data = split_data.stack()
        if index is not None:
            stacked_data = stacked_data.set_axis(index)
        else:
            stacked_data = stacked_data.reset_index(drop=True)
    else:
        subplots_num = cluster_count

    with plt.rc_context(rc={"figure.max_open_warning": 0}):
        with plt.style.context(style):
            fig = plt.figure(figsize=(15, subplots_num*5), constrained_layout=True)

            # cluster center plots
            ax = []
            for cluster in range(cluster_count):
                ax.append(fig.add_subplot(subplots_num, 1, cluster + 1))
                # plot clustered data if available
                if split_data is not None:
                    ax[cluster].plot(split_data[data[column_name] == cluster].T, c=cluster_colors[cluster], alpha=0.05)
                ax[cluster].plot(model.cluster_centers_[cluster], c=cluster_colors[cluster], linewidth=5)
                if split_data is not None:
                    ax[cluster].set_title("Cluster {cluster} Plot with Cluster Center and Clustered Samples"
                                          .format(cluster=cluster),
                                          fontsize=17.5, fontweight="bold", pad=10)
                else:
                    ax[cluster].set_title("Cluster {cluster} Plot with Cluster Center"
                                          .format(cluster=cluster),
                                          fontsize=17.5, fontweight="bold", pad=10)
                ax[cluster].set_xlabel("Split", fontsize=15)
                ax[cluster].set_ylabel("Value", fontsize=15)

            if split_data is not None:
                # cluster center time plot
                ax.append(fig.add_subplot(subplots_num, 1, subplots_num))
                ax[cluster_count].plot(stacked_data, c='black', alpha=0.3)
                for i in range(len(split_data.columns)):
                    ax[cluster_count].scatter(split_data.index, split_data[split_data.columns[i]],
                                              c=[cluster_colors[cluster] for cluster in data[column_name]])
                # only show legend for up to 15 clusters
                if cluster_count <= 15:
                    handles = []
                    for cluster in range(cluster_count):
                        handles.append(mpatches.Patch(color=cluster_colors[cluster], label=cluster))
                    ax[cluster_count].legend(handles=handles,
                                             title="Cluster", loc="upper right", bbox_to_anchor=(1, 1),
                                             frameon=True, facecolor="white")
                    ax[cluster_count].set_title("Scatter Plot with Clustered Samples on Data Over Time",
                                                fontsize=17.5, fontweight="bold", pad=10)
                ax[cluster_count].xaxis.set_tick_params(labelrotation=60)
                ax[cluster_count].set_xlabel("Year", fontsize=15)
                ax[cluster_count].set_ylabel("Value", fontsize=15)

            # format figure
            if description != "":
                fig.suptitle("{column_name}: {description}".format(column_name=column_name, description=description),
                             fontsize=25, fontweight="bold")
            else:
                fig.suptitle("{column_name}".format(column_name=column_name),
                             fontsize=25, fontweight="bold")

            fig.set_constrained_layout_pads(h_pad=0.3, w_pad=0.3, hspace=0.1)

            if display:
                fig.show()

    if get:
        return fig
    else:
        plt.close(fig)
        return None


def make_lowess_plot(column, smooth_column, data=None, description="LOWESS", display=True, get=False,
                     style="seaborn-darkgrid"):
    """
    Makes plot figure to visualize data smoothed with LOWESS (Locally Weighted Scatterplot Smoothing).

    Figure includes a plot with a scatter plot for values before smoothing and a line plot for values after smoothing.

    :param column: value Series or column name (requires data parameter to get value Series)
    :param smooth_column: smooth value Series or smooth column name (requires data parameter to get value Series)
    :param data: data DataFrame with columns of column and smooth_column parameters included (used if column or
                 smooth_column parameter are names)
    :param description: description to add to plot figure title after column name, if "" add no description
    :param display: display plot figure if True (disable plot window to avoid errors when plot limit is reached)
    :param get: return plot figure if True, otherwise close created plot figure and return None
    :param style: style name for plot figure
    :return: plot figure if get parameter is True or None if get parameter is False or required column data is not given
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

    smooth_column_type = utils.get_data_label(smooth_column)
    if smooth_column_type in {"str", "int"}:
        if data is not None:
            try:
                smooth_column = data[smooth_column]
            except KeyError:
                return None
        else:
            return None
    elif smooth_column_type != "series":
        return None

    with plt.rc_context(rc={"figure.max_open_warning": 0}):
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=(15, 5))

            # before and after LOWESS
            ax.scatter(x=column.index, y=column, cmap="tab20")  # before smoothing
            ax.plot(smooth_column.index, smooth_column, c="k")  # after smoothing
            ax.set_title("Scatter Plot for Values Before Smoothing and Line Plot for Values After Smoothing",
                         fontsize=17.5, fontweight="bold", pad=10)
            ax.set_xlabel("Index", fontsize=15)
            ax.set_ylabel("Value", fontsize=15)

            # format figure
            if description != "":
                fig.suptitle("{column_name}: {description}".format(column_name=column_name, description=description),
                             fontsize=25, fontweight="bold")
            else:
                fig.suptitle("{column_name}".format(column_name=column_name),
                             fontsize=25, fontweight="bold")
            fig.tight_layout(pad=1.5)

            if display:
                fig.show()

    if get:
        return fig
    else:
        plt.close(fig)
        return None


def make_distribution_plot(column, data=None, description="", display=True, get=False, style="seaborn-darkgrid"):
    """
    Makes plot figure to visualize data distribution.

    Figure includes a count plot for missing values, a histogram plot, and a box plot (if column is numeric).

    :param column: value Series or column name (requires data parameter to get value Series)
    :param data: data DataFrame with column of column parameter included (used if column parameter is name)
    :param description: description to add to plot figure title after column name, if "" add no description
    :param display: display plot figure if True (disable plot window to avoid errors when plot limit is reached)
    :param get: return plot figure if True, otherwise close created plot figure and return None
    :param style: style name for plot figure
    :return: plot figure if get parameter is True or None if get parameter is False or required column data is not given
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

    # add column for missing values (True if missing, False if present)
    column_data = pd.concat([column, pd.Series(pd.isna(column), name="missing_value")], axis=1)

    has_numeric = is_numeric_dtype(column_data[column_name])
    has_low_unique = column_data[column_name].nunique() <= 50
    # select figure options
    if has_numeric:
        fig_height = 15
        subplots_num = 3
        subplots_adjust_top = 0.9
    elif has_low_unique:
        fig_height = 10
        subplots_num = 2
        subplots_adjust_top = 0.86
    else:
        fig_height = 5
        subplots_num = 1
        subplots_adjust_top = 0.82

    with plt.rc_context(rc={"figure.max_open_warning": 0}):
        with plt.style.context(style):
            fig = plt.figure(figsize=(15, fig_height))

            # count plot
            ax1 = fig.add_subplot(subplots_num, 1, 1)
            sns.countplot(data=column_data, y="missing_value", ax=ax1,
                          palette={False: "C0", True: "C1"})
            #for container in ax1.containers:
            #    ax1.bar_label(container)
            ax1.set_title("Count Plot for Missing Data",
                          fontsize=17.5, fontweight="bold", pad=10)
            ax1.set_xlabel("Count", fontsize=15)
            ax1.set_ylabel("Missing Values", fontsize=15)

            # histogram plot
            if has_numeric or has_low_unique:
                ax2 = fig.add_subplot(subplots_num, 1, 2)
                sns.histplot(data=column_data, x=column_name, ax=ax2,
                             stat="count", kde=True)
                ax2.set_title("Histogram Plot for Data Distribution (with KDE)",
                              fontsize=17.5, fontweight="bold", pad=10)
                if not has_numeric:
                    ax2.xaxis.set_tick_params(labelrotation=60)
                ax2.set_xlabel("Value", fontsize=15)
                ax2.set_ylabel("Count", fontsize=15)

            # box plot
            if has_numeric:
                ax3 = fig.add_subplot(subplots_num, 1, 3)
                sns.boxplot(data=column_data, x=column_name, ax=ax3,
                            showmeans=True,
                            meanprops={"marker": "o",
                                       "markerfacecolor": "white",
                                       "markeredgecolor": "black",
                                       "markersize": "12"})
                ax3.set_title("Box Plot for Data Distribution (IQR, Median, Mean, Whiskers, Outliers)",
                              fontsize=17.5, fontweight="bold", pad=10)
                ax3.set_xlabel("Value", fontsize=15)

            # format figure
            if description != "":
                fig.suptitle("{column_name}: {description}".format(column_name=column_name, description=description),
                             fontsize=25, fontweight="bold")
            else:
                fig.suptitle("{column_name}".format(column_name=column_name),
                             fontsize=25, fontweight="bold")
            fig.tight_layout(pad=1.5)
            fig.subplots_adjust(top=subplots_adjust_top, hspace=0.3)

            if display:
                fig.show()

    if get:
        return fig
    else:
        plt.close(fig)
        return None


def make_time_plot(column, data=None, description="", display=True, get=False, style="seaborn-darkgrid"):
    """
    Makes plot figure to visualize data trend and seasonality.

    Figure includes a line plot for data over time, a line plot and a scatter plot for monthly data, and grouped plots
    for monthly and quarterly data (if downsampled data doesn't have missing values and all months are available).

    :param column: value Series with Timestamp index or column name (requires data parameter to get value Series)
    :param data: data DataFrame with column of column parameter included (used if column parameter is name)
    :param description: description to add to plot figure title after column name, if "" add no description
    :param display: display plot figure if True (disable plot window to avoid errors when plot limit is reached)
    :param get: return plot figure if True, otherwise close created plot figure and return None
    :param style: style name for plot figure
    :return: plot figure if get parameter is True or None if get parameter is False or required column data is not
             given, not numeric, or has only missing values or column index isn't only dates
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
    if not is_numeric_dtype(column) or column.isna().all() or not isinstance(column.index, pd.DatetimeIndex):
        return None
    column_name = column.name

    # downsample data to monthly data and drop rows with nan values
    column_data = column.resample("MS").mean().to_frame()
    has_nan = column_data[column_name].isna().any()
    if has_nan:
        column_data = column_data.dropna()
    column_data["month"] = column_data.index.month_name().str.slice(stop=3)
    column_data["year"] = column_data.index.year

    year_count = column_data["year"].nunique()
    month_count = column_data["month"].nunique()
    has_one_year = year_count == 1
    has_missing_months = month_count != 12

    # map months to angles for polar plot
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    angles = np.linspace(0, 2*np.pi, len(months), endpoint=False)
    month_angle_dict = dict(zip(months, angles))
    column_data["angle"] = column_data["month"].map(month_angle_dict)

    # select figure options
    if has_nan or has_missing_months or has_one_year:
        fig_height = 15
        subplots_num = 3
        subplots_adjust_top = 0.9
    else:
        fig_height = 30
        subplots_num = 5
        subplots_adjust_top = 0.94

    with plt.rc_context(rc={"figure.max_open_warning": 0}):
        with plt.style.context(style):
            fig = plt.figure(figsize=(15, fig_height))

            # time line plot
            ax1 = fig.add_subplot(subplots_num, 1, 1)
            sns.lineplot(data=column_data, x=column_data.index, y=column_name, ax=ax1)
            ax1.set_title("Line Plot for Data Over Time (Monthly Samples)",
                          fontsize=17.5, fontweight="bold", pad=10)
            # set yearly major ticks
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            #x_ticks = ax1.xaxis.get_major_ticks()
            #x_ticks[1].set_visible(False)
            #x_ticks[-1].set_visible(False)
            ax1.xaxis.set_tick_params(labelrotation=60)
            ax1.set_xlabel("Year", fontsize=15)
            ax1.set_ylabel("Value", fontsize=15)

            # seasonal line plot (monthly)
            ax2 = fig.add_subplot(subplots_num, 1, 2)
            sns.lineplot(data=column_data, x="month", y=column_name, hue="year", ax=ax2,
                         legend="full", palette="tab20")
            # only show legend for up to 20 years
            if year_count <= 20:
                ax2.legend(title="Year", loc="upper right", bbox_to_anchor=(1.003, 1),
                           frameon=True, facecolor="white")
            else:
                ax2.legend().set_visible(False)
            ax2.set_title("Seasonal Line Plot for Data Trend and Seasonality (Monthly Samples)",
                          fontsize=17.5, fontweight="bold", pad=10)
            ax2.set_xlabel("Month", fontsize=15)
            ax2.set_ylabel("Value", fontsize=15)

            # polar seasonal scatter plot (monthly)
            ax3 = fig.add_subplot(subplots_num, 1, 3, projection="polar")
            scatter = ax3.scatter(x=column_data["angle"], y=column_data[column_name],
                                  c=column_data["year"], cmap="tab20")
            ax3.set_thetagrids(angles * 180 / np.pi, months)
            # only show legend for up to 20 years
            if year_count <= 20:
                handles, labels = scatter.legend_elements(num=None)
                ax3.legend(handles=handles, labels=labels,
                           title="Year", loc="upper right", bbox_to_anchor=(1.4, 1),
                           frameon=True, facecolor="white")
            ax3.set_title("Polar Seasonal Scatter Plot for Data Trend and Seasonality (Monthly Samples)",
                          fontsize=17.5, fontweight="bold", pad=10)

            if not has_nan and not has_missing_months and not has_one_year:
                # grouped seasonal plot (monthly)
                ax4 = fig.add_subplot(subplots_num, 1, 4)
                month_plot(x=column_data[column_name], ylabel=column_name, ax=ax4)
                ax4.set_title("Grouped Seasonal Plot for Data Trend and Seasonality (Monthly Samples)",
                              fontsize=17.5, fontweight="bold", pad=10)
                ax4.set_xlabel("Month", fontsize=15)
                ax4.set_ylabel("Value", fontsize=15)

                # grouped seasonal plot (quarterly)
                ax5 = fig.add_subplot(subplots_num, 1, 5)
                quarter_plot(x=column.resample("QS").mean(), ylabel=column_name, ax=ax5)
                ax5.set_title("Grouped Seasonal Plot for Data Trend and Seasonality (Quarterly Samples)",
                              fontsize=17.5, fontweight="bold", pad=10)
                ax5.set_xlabel("Quarter", fontsize=15)
                ax5.set_ylabel("Value", fontsize=15)

            # format figure
            if description != "":
                fig.suptitle("{column_name}: {description}".format(column_name=column_name, description=description),
                             fontsize=25, fontweight="bold")
            else:
                fig.suptitle("{column_name}".format(column_name=column_name),
                             fontsize=25, fontweight="bold")
            fig.tight_layout(pad=1.5)
            fig.subplots_adjust(top=subplots_adjust_top, hspace=0.3)

            if display:
                fig.show()

    if get:
        return fig
    else:
        plt.close(fig)
        return None


def make_data_plots(data, distribution_plots=True, time_plots=True, descriptions=None, display=True, get=False,
                    style="seaborn-darkgrid"):
    """
    Makes plot figures for data columns to visualize data.

    :param data: data DataFrame
    :param distribution_plots: make distribution plot figures if True
    :param time_plots: make time (trend and seasonality) plot figures if True
    :param descriptions: dictionary with column descriptions to add to plot figure titles after column names, if None
                         add no descriptions
    :param display: display plot figures if True (disable plot window to avoid errors when plot limit is reached)
    :param get: return plot figures if True, otherwise close created plot figures and return None
    :param style: style name for plot figures
    :return: plot figures dictionary of dictionaries (column plot figure for each plot type), plot figures dictionary
             if only 1 type of plots is selected, or None if get parameter is False
    """
    if distribution_plots:
        data_distribution_plots = dict()
    if time_plots:
        data_time_plots = dict()
    # plot for every column
    for column in data.columns:
        if descriptions is not None:
            description = descriptions.get(column, "")
        else:
            description = ""

        if distribution_plots:
            data_distribution_plots[column] = make_distribution_plot(data[column], description=description,
                                                                     display=display, get=get, style=style)

        if time_plots:
            data_time_plots[column] = make_time_plot(data[column], description=description,
                                                     display=display, get=get, style=style)

    if get:
        # column plot figure for each plot type
        plots = dict()
        if data_distribution_plots:
            plots["distribution"] = data_distribution_plots
        if data_time_plots:
            plots["time"] = data_time_plots

        if len(plots) == 1:
            return list(plots.items())[0][1]
        else:
            return plots
    else:
        return None


def make_and_output_station_data_plots(start_datetime, end_datetime, stations, category="fetched", subcategory="",
                                       main_columns=False, display=False):
    """
    Makes station data plot figures and outputs them to files.

    :param start_datetime: start datetime for data as Timestamp object
    :param end_datetime: end datetime for data as Timestamp object
    :param stations: station ID, station ID list, or station info DataFrame
    :param category: select kind of data being plotted -
                     fetched data if "fetched" (make distribution and time plot figures for all features selected
                     with main_columns parameter),
                     preprocessed data if "preprocessed" (make plot figures according to subcategory parameter),
                     modeled data if "modeled" (make plot figures according to subcategory parameter)
    :param subcategory: used with category parameter to select kind of data being plotted -
                        if category parameter is "preprocessed" -
                        clean data if "clean" or "" (make distribution and time data plot figures for all features),
                        clean data scaled with MinMaxScaler if "scaled_minmax" (make distribution and time plot figures
                        for all features selected with main_columns parameter),
                        clean data scaled with StandardScaler if "scaled_standard" (make distribution and time plot
                        figures for all features selected with main_columns parameter)
                        if category parameter is "modeled" -
                        split clean data clustered using K-Means clustering with DTW if "kmeans_timeseries" (make time
                        series cluster plot figures for all features)
    :param main_columns: plot only temperature, dew point temperature, relative humidity, wind speed, sky level 1
                         coverage code, sky level 1 altitude, and apparent temperature columns if True otherwise
                         additionally plot wind direction, precipitation, visibility, wind gust, sky level 2-3 coverage
                         codes, and sky level 2-3 altitude columns (if they exist in read data file, ignored if category
                         parameter is not "fetched" or "preprocessed")
    :param display: display plot figures if True (disable plot window to avoid errors when plot limit is reached)
    """
    if utils.get_data_label(stations) == "list":
        stations = utils.read_station_info(stations=stations, main_columns=True)
    if stations is None:
        return

    for sid, network in zip(stations["sid"], stations["network"]):
        if category in {"fetched", "preprocessed"}:
            if category == "fetched":
                subcategory = ""
            if category == "preprocessed" and subcategory == "":
                subcategory = "clean"

            # input data
            station_data = utils.read_station_data(sid, start_datetime, end_datetime,
                                                   network=network, category=category, subcategory=subcategory,
                                                   main_columns=main_columns)
            # no data for station
            if station_data is None:
                continue

            # distribution and time plots for every column
            dist_directory_out = utils.join_path("plots", network, sid, category, subcategory, "distribution",
                                                 root=True)
            time_directory_out = utils.join_path("plots", network, sid, category, subcategory, "time", root=True)
            # column descriptions for plots
            descriptions = utils.get_station_data_descriptions(preprocessed=category == "preprocessed",
                                                               units=subcategory in {"", "clean"})
            for column in station_data.columns:
                file_out = utils.join_file(sid, start_datetime, end_datetime, column, sep="_", ext=".png", convert=True)

                # distribution plot
                dist_plot = make_distribution_plot(column, station_data, description=descriptions.get(column, ""),
                                                   display=display, get=True)
                # output distribution plot
                if dist_plot is not None:
                    utils.write_data_to_file(dist_plot, file_out, dist_directory_out)
                    plt.close(dist_plot)

                # time plot
                time_plot = make_time_plot(column, station_data, description=descriptions.get(column, ""),
                                           display=display, get=True)
                # output time plot
                if time_plot is not None:
                    utils.write_data_to_file(time_plot, file_out, time_directory_out)
                    plt.close(dist_plot)

            """
            # make all plots
            plots = make_data_plots(station_data, distribution_plots=True, time_plots=True,
                                    descriptions=descriptions, display=display, get=True)

            dist_directory_out = utils.join_path("plots", network, sid, category, "distribution", root=True)
            time_directory_out = utils.join_path("plots", network, sid, category, "time", root=True)
            for column in station_data.columns:
                file_out = utils.join_file(sid, start_datetime, end_datetime, column, sep="_", ext=".png", convert=True)
                dist_plot = plots["distribution"][column]
                time_plot = plots["time"][column]
                if dist_plot is not None:
                    utils.write_data_to_file(dist_plot, file_out, dist_directory_out)
                    plt.close(dist_plot)
                if time_plot is not None:
                    utils.write_data_to_file(time_plot, file_out, time_directory_out)
                    plt.close(time_plot)
            """
        elif category == "modeled":
            if subcategory == "kmeans_timeseries":
                # input time series clustered data
                clustered_data = utils.read_station_data(sid, start_datetime, end_datetime,
                                                         network=network, category=category, subcategory=subcategory)
                # no data for station
                if clustered_data is None:
                    continue

                # time series cluster plots for every column
                tscluster_directory_out = utils.join_path("plots", network, sid, category, subcategory, root=True)
                descriptions = utils.get_station_data_descriptions(preprocessed=True)  # column descriptions for plots
                for column in clustered_data.columns:
                    # input split preprocessed data for column
                    file_in = utils.join_file(sid, start_datetime, end_datetime, column, sep="_", ext=".csv",
                                              convert=True)
                    directory_in = utils.join_path("data", network, sid, "preprocessed", "clean", root=True)
                    split_column_data = utils.read_data_from_file(file_in, directory_in, index_col="valid",
                                                                  parse_dates=True)
                    # no data for column
                    if split_column_data is None:
                        continue

                    # input time series clustering model
                    model_file = utils.join_file(sid, column, sep="_", ext=".json")
                    model_directory = utils.join_path("models", network, sid, "kmeans_timeseries", root=True)
                    model_object = utils.read_data_from_file(model_file, model_directory, reading="tslearn",
                                                             type="json", mclass=TimeSeriesKMeans)
                    # no model for column
                    if model_object is None:
                        continue

                    # time series cluster plot
                    tscluster_plot = make_timeseries_cluster_plot(column, model=model_object,
                                                                  data=clustered_data, split_data=split_column_data,
                                                                  index=pd.date_range(start=start_datetime,
                                                                                      end=end_datetime,
                                                                                      freq="H", inclusive="left"),
                                                                  description=descriptions.get(column, ""),
                                                                  display=display, get=True)

                    # output time series cluster plot
                    if tscluster_plot is not None:
                        tscluster_file_out = utils.join_file(sid, start_datetime, end_datetime, column, sep="_",
                                                             ext=".png", convert=True)
                        utils.write_data_to_file(tscluster_plot, tscluster_file_out, tscluster_directory_out)
                        plt.close(tscluster_plot)
