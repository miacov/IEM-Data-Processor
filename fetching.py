import pandas as pd
import bs4 as bs
from urllib.request import urlopen
import json
import time
from io import StringIO
import utils


def fetch_network_info():
    """
    Fetches network info from the IEM.

    :return: network info DataFrame
    """
    source = urlopen("https://www.mesonet.agron.iastate.edu/request/download.phtml").read()
    soup = bs.BeautifulSoup(source, "html.parser")
    options = soup.find("form", {"name": "netselect"}).find_all("option")
    network_info = []
    for option in options:
        network_info.append({"network": option["value"],
                             "network_name": option.get_text()})

    return pd.DataFrame(network_info)


def fetch_and_output_network_info():
    """
    Fetches network info from the IEM and outputs it to file.
    """
    network_info = fetch_network_info()
    utils.write_network_info(network_info)


def fetch_station_info(network, extra_columns=False, geojson=False):
    """
    Fetches station info from an ASOS network in the IEM.

    :param network: network code
    :param extra_columns: include all columns if True, otherwise drop extra columns
    :param geojson: station info as GeoJSON dictionary if True, otherwise DataFrame
    :return: station info DataFrame or GeoJSON dictionary or None if fetching failed
    """
    source = urlopen("https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson".format(network=network))
    station_info_dict = json.load(source)

    station_info_dicts = []
    for station in station_info_dict["features"]:
        station_info_cur = station["properties"]

        # delete columns
        if not extra_columns:
            del station_info_cur["climate_site"]
            del station_info_cur["wfo"]
            #del station_info_cur["tzname"]
            del station_info_cur["ncdc81"]
            del station_info_cur["ncei91"]
            del station_info_cur["ugc_county"]
            del station_info_cur["ugc_zone"]
            #del station_info_cur["county"]

        # format station info
        time_domain = station_info_cur["time_domain"].replace("(", "").replace(")", "").split("-")
        del station_info_cur["time_domain"]
        station_info_cur["start"] = time_domain[0]
        station_info_cur["end"] = time_domain[1].replace("Now", "")

        station_info_cur["sname"] = station_info_cur["sname"].replace("  ", " ")
        if pd.notna(station_info_cur["state"]):
            station_info_cur["state"] = station_info_cur["state"].replace(" ", "")
        if pd.notna(station_info_cur["country"]):
            station_info_cur["country"] = station_info_cur["country"].replace(" ", "")

        # dataframe formatting
        if not geojson:
            station_info_cur["lat"] = station["geometry"]["coordinates"][1]
            station_info_cur["lon"] = station["geometry"]["coordinates"][0]
            station_info_dicts.append(station_info_cur)

    if geojson:
        if station_info_dict["count"] != 0:
            return station_info_dict
        else:
            return None
    else:
        station_info_df = pd.DataFrame(station_info_dicts)
        if not station_info_df.empty:
            return station_info_df
        else:
            return None


def fetch_and_output_station_info(networks):
    """
    Fetches station info from ASOS networks in the IEM and outputs it to file.

    :param networks: network code, network code list, or network info DataFrame
    """
    item_type = utils.get_data_label(networks)
    if item_type == "df":
        networks = networks["network"].tolist()
    elif item_type == "str":
        networks = [networks]

    station_info = None
    for network in networks:
        station_info_cur = fetch_station_info(network)

        if station_info_cur is not None:
            # station info from every network to one df
            if station_info is None:
                station_info = station_info_cur
            else:
                station_info = pd.concat([station_info, station_info_cur], ignore_index=True)

    if station_info is not None:
        utils.write_station_info(station_info)


def get_service(start_datetime, end_datetime, report="", timezone="Etc/UTC", trace=False, latlon=False,
                elevation=False):
    """
    Makes a URI string with selected service options to use for data fetching.

    :param start_datetime: start datetime for data as Timestamp object
    :param end_datetime: end datetime for data as Timestamp object
    :param report: limit report types -
                   MADIS HFMETAR (5 minute ASOS) if "frequent",
                   routine (once hourly) if "routine",
                   specials if "special",
                   routine and specials if "combined",
                   all reports otherwise
    :param timezone: timezone for presentation of observation times
    :param trace: trace reports represented with "0.0001" if True, otherwise represented with empty string
    :param latlon: include station latitude and longitude as columns in reports
    :param elevation: include station elevation as column in reports
    :return: service URI string with fetching options
    """
    service = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    service += "data=all&"  # get all available data
    # report type, default all
    if report == "frequent":
        service += "report_type=1&"  # 5 minute
    elif report == "combined":
        service += "report_type=2&"  # routine + special
    elif report == "routine":
        service += "report_type=3&"  # routine
    elif report == "special":
        service += "report_type=4&"  # special

    # formatting
    service += "tz={timezone}&".format(timezone=timezone)  # time, default UTC
    service += "format=onlycomma&"  # comma delimited, no debug headers
    service += "missing=empty&"  # missing data represented with empty string
    if not trace:
        service += "trace=empty&"  # trace reports represented with empty string

    # station info
    if latlon:
        service += "latlon=yes&"  # include latitude and longitude
    if elevation:
        service += "elev=yes&"  # include elevation

    # datetimes
    service += start_datetime.strftime("year1=%Y&month1=%m&day1=%d&hour1=%H&minute1=%M&")
    service += end_datetime.strftime("year2=%Y&month2=%m&day2=%d&hour2=%H&minute2=%M&")

    return service


def fetch_station_data(service, station="", duplicates=False, extra_columns=False, attempts=5, timeout=120, sleep=5):
    """
    Fetches station data from the IEM.

    :param service: service URI string with fetching options
    :param station: station id, if "" fetch data from all stations up to a period of 24 hours
    :param duplicates: allow samples from different report types to have the same timestamp if True, otherwise drop the
                       samples that have more nan values than their duplicates
    :param extra_columns: include all columns if True, otherwise drop extra columns (METAR and snowdepth)
    :param attempts: max number of attempts to download data
    :param timeout: time in seconds before timeout for fetching
    :param sleep: time in seconds to sleep after fetching fail
    :return: station data DataFrame or None if fetching failed
    """
    if not station == "":
        service = "{service}&station={sid}".format(service=service, sid=station)  # add station to uri

    attempt = 0
    while attempt < attempts:
        try:
            source = urlopen(service, timeout=timeout).read().decode("utf-8")
            if source is not None and not source.startswith("ERROR"):
                # format station data
                station_data = pd.read_csv(StringIO(source), sep=",", dtype=utils.get_station_data_dtypes())

                # drop columns
                # METAR: info split into other variables
                # snowdepth: missing info
                if not extra_columns:
                    station_data = station_data.drop(["metar", "snowdepth"], axis=1, errors="ignore")

                # drop duplicates with more nan values
                if not duplicates:
                    station_data["non_nan_columns"] = station_data.count(axis=1)
                    station_data = (
                        station_data.sort_values(["valid", "non_nan_columns"])
                                    .drop_duplicates("valid", keep="last")
                                    .drop("non_nan_columns", axis=1)
                    )

                return station_data
        except Exception:
            time.sleep(sleep)
        attempt += 1

    # data fetching failed
    return None


def fetch_and_output_station_data(start_datetime, end_datetime, stations=None, report="", timezone="Etc/UTC",
                                  trace=False):
    """
    Fetches station data from the IEM and outputs it to files.

    :param start_datetime: start datetime for data as Timestamp object
    :param end_datetime: end datetime for data as Timestamp object
    :param stations: station ID, station ID list, or station info DataFrame to fetch data from each station in a
                     separate file, or None to fetch data from all stations up to a period of 24 hours in one file
                     (end_datetime parameter up to 24 hours after start_datetime parameter)
    :param report: limit report types -
                   MADIS HFMETAR (5 minute ASOS) if "frequent",
                   routine (once hourly) if "routine",
                   specials if "special",
                   routine and specials if "combined",
                   all reports otherwise
    :param timezone: timezone for presentation of observation times
    :param trace: trace reports represented with "0.0001" if True, otherwise represented with empty string
    """
    if utils.get_data_label(stations) == "list":
        stations = utils.read_station_info(stations=stations, main_columns=True)

    service = get_service(start_datetime, end_datetime, report=report, timezone=timezone, trace=trace)

    if stations is not None:
        # fetch data from each station separately
        for sid, network in zip(stations["sid"], stations["network"]):
            station_data = fetch_station_data(service, sid)

            # output fetched data
            if station_data is not None:
                utils.write_station_data(station_data, sid, start_datetime, end_datetime,
                                         network=network, category="fetched", index_column=False)
    else:
        # fetch data from all stations up to 24 hours
        if (end_datetime - start_datetime).total_seconds() / (60 * 60 * 24) <= 1:
            station_data = fetch_station_data(service)

            # output fetched data
            if station_data is not None:
                utils.write_station_data(station_data, "ALL", start_datetime, end_datetime,
                                         network="ALL", category="fetched", index_column=False)
