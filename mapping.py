import pandas as pd
import folium
import math
import utm
from html2image import Html2Image
import utils


def convert_utm_to_latlon(zone_number, column_x, column_y, data=None, zone_letter="", northern=True):
    """
    Converts UTM (Universal Transverse Mercator) x (easting) and y (northing) coordinates to latitude and longitude.

    :param zone_number: zone number (1-60)
    :param column_x: x coordinate Series or x column name (requires data parameter to get value Series)
    :param column_y: y coordinate Series or y column name (requires data parameter to get value Series)
    :param data: data DataFrame with columns of column_x and column_y parameters included (used if column_x or column_y
                 parameter are names)
    :param zone_letter: zone letter (uses northern parameter if "")
    :param northern: northern hemisphere if True (ignored if zone_letter parameter is not "")
    :return: converted latitude and longitude DataFrame if column_x and column_y parameter are Series, data parameter
             DataFrame with converted latitude and longitude columns added and previous columns dropped if data
             parameter is not None, or None if converting failed
    """
    column_type_x = utils.get_data_label(column_x)
    if column_type_x in {"str", "int"}:
        if data is not None:
            try:
                column_x = data[column_x]
            except KeyError:
                return None
        else:
            return None
    elif column_type_x != "series":
        return None
    column_x_name = column_x.name

    column_type_y = utils.get_data_label(column_y)
    if column_type_y in {"str", "int"}:
        if data is not None:
            try:
                column_y = data[column_y]
            except KeyError:
                return None
        else:
            return None
    elif column_type_y != "series":
        return None
    column_y_name = column_y.name

    try:
        if zone_letter != "":
            lat, lon = utm.to_latlon(column_x, column_y, zone_number=zone_number, zone_letter=zone_letter)
        else:
            lat, lon = utm.to_latlon(column_x, column_y, zone_number=zone_number, northern=northern)
    except Exception:
        return None

    column_lat = pd.Series(lat, name="lat")
    column_lon = pd.Series(lon, name="lon")
    latlon_data = pd.concat([column_lat, column_lon], axis=1)

    if data is not None:
        data = data.drop([column_x_name, column_y_name], axis=1, errors="ignore")
        latlon_data = pd.concat([data, latlon_data], axis=1)  # add to existing dataframe

    return latlon_data


def get_html_table(info, datetime_to_year=False):
    """
    Makes an HTML table string from info values.

    :param info: Series with info values
    :param datetime_to_year: convert datetime info Series values to year values if True
    :return: HTML table string
    """
    # format series
    info = info.to_frame()
    info = info[~info[info.columns[0]].isna()]  # remove missing
    if datetime_to_year:
        for row in info.itertuples():
            if utils.get_data_label(row[1]) == "datetime":
                info.loc[row[0], info.columns[0]] = info.loc[row[0], info.columns[0]].year  # datetime to year only
    info.index = info.index.str.capitalize()  # capitalize first letter of index values

    # format html
    html_table = info.to_html(header=False, bold_rows=True)
    html_table = html_table.replace("Sname", "Name").replace("Tzname", "Timezone").replace("Sid", "ID")
    html_table = html_table.replace("Lat", "Latitude")
    html_table = html_table.replace("Lon", "Longitude").replace("Long", "Longitude").replace("Lng", "Longitude")
    html_table = html_table.replace("<th>", '<th style="text-align:left">')  # format header cells
    html_table = html_table.replace("<td>", '<td style="text-align:center">')  # format cells

    return html_table


def draw_circlemarker(info, group, datetime_to_year=False):
    """
    Draws a circle marker overlay with an info popup.

    :param info: Series with values for marker center (latitude as "lat", "latitude" and longitude as "lon", "long",
                 "lng", or "longitude") and additional info for popup
    :param group: folium layer overlay or Map object to add marker to
    :param datetime_to_year: convert datetime info Series values to year values if True
    :return:
    """
    if "lat" in info.index:
        lat = info["lat"]
    elif "latitude" in info.index:
        lat = info["latitude"]
    else:
        return

    if "lon" in info.index:
        lon = info["lon"]
    elif "long" in info.index:
        lon = info["long"]
    elif "lng" in info.index:
        lon = info["lng"]
    elif "longitude" in info.index:
        lon = info["longitude"]
    else:
        return

    try:
        folium.CircleMarker(location=(lat, lon),
                            radius=5,
                            color="darkblue",
                            weight=4,
                            fill=True,
                            fill_color="lightblue",
                            fill_opacity=1,
                            popup=get_html_table(info, datetime_to_year)).add_to(group)
    except ValueError:
        return


def draw_circle(info, group, datetime_to_year=False, spread_as_radius=False):
    """
    Draws a circle overlay with an info popup.

    :param info: Series with values for circle center (latitude as "lat", "latitude" and longitude as "lon", "long",
                 "lng", or "longitude"), radius ("radius" or "spread" if spread_as_radius if parameter is True), and
                 additional info for popup
    :param group: folium layer overlay or Map object to add circle to
    :param datetime_to_year: convert datetime info Series values to year values if True
    :param spread_as_radius: use "spread" Series value converted from hectares to a radius for circle radius if True,
                             otherwise use "radius" value (if selected value is missing then radius is set to 1 meter)
    """
    if "lat" in info.index:
        lat = info["lat"]
    elif "latitude" in info.index:
        lat = info["latitude"]
    else:
        return

    if "lon" in info.index:
        lon = info["lon"]
    elif "long" in info.index:
        lon = info["long"]
    elif "lng" in info.index:
        lon = info["lng"]
    elif "longitude" in info.index:
        lon = info["longitude"]
    else:
        return

    if spread_as_radius:
        try:
            radius = math.sqrt(float(info["spread"])*10000)/2  # hectares to m2 to radius
        except (ValueError, KeyError):
            radius = 1  # small circle if spread is missing (1m)
    else:
        try:
            radius = float(info["radius"])
        except (ValueError, KeyError):
            radius = 1  # small circle if radius is missing (1m)

    try:
        folium.Circle(location=(lat, lon),
                      radius=radius,
                      color="darkred",
                      weight=3,
                      fill=True,
                      fill_color="red",
                      fill_opacity=0.1,
                      popup=get_html_table(info, datetime_to_year)).add_to(group)
    except ValueError:
        return


def draw_polyline(info, group, suffixes=None):
    """
    Draws a line overlay between two points.

    :param info: Series with values for point coordinates (latitude as "lat", "latitude" and longitude as "lon", "long",
                 "lng", or "longitude" with suffixes if suffixes parameter is not None)
    :param group: folium layer overlay or Map object
    :param suffixes: suffixes turple or list to differentiate point columns or None to use first 2 info Series values
                     for point 1 and last 2 values for point 2
    """
    try:
        if suffixes is None:
            point_from = (info.iloc[0], info.iloc[1])
            point_to = (info.iloc[2], info.iloc[4])
        else:
            if "lat" + suffixes[0] in info.index:
                lat_from = info["lat" + suffixes[0]]
            elif "latitude" in info.index:
                lat_from = info["latitude" + suffixes[0]]
            else:
                return

            if "lat" + suffixes[1] in info.index:
                lat_to = info["lat" + suffixes[1]]
            elif "latitude" in info.index:
                lat_to = info["latitude" + suffixes[1]]
            else:
                return

            if "lon" + suffixes[0] in info.index:
                lon_from = info["lon" + suffixes[0]]
            elif "long" + suffixes[0] in info.index:
                lon_from = info["long" + suffixes[0]]
            elif "lng" + suffixes[0] in info.index:
                lon_from = info["lng" + suffixes[0]]
            elif "longitude" + suffixes[0] in info.index:
                lon_from = info["longitude" + suffixes[0]]
            else:
                return

            if "lon" + suffixes[1] in info.index:
                lon_to = info["lon" + suffixes[1]]
            elif "long" + suffixes[1] in info.index:
                lon_to = info["long" + suffixes[1]]
            elif "lng" + suffixes[1] in info.index:
                lon_to = info["lng" + suffixes[1]]
            elif "longitude" + suffixes[1] in info.index:
                lon_to = info["longitude" + suffixes[1]]
            else:
                return

            point_from = (lat_from, lon_from)
            point_to = (lat_to, lon_to)
    except (ValueError, IndexError):
        return

    folium.PolyLine(locations=[point_from, point_to],
                    color="purple",
                    weight=1).add_to(group)


def make_yearly_fire_info_map(fire_info):
    """
    Makes map to visualize fire locations for every year.

    Map includes fire location circles with spread.

    :param fire_info: fire info DataFrame
    :return: map or None if fire_info parameter is None or empty
    """
    if fire_info is not None:
        if fire_info.empty:
            return None
    else:
        return None

    # fire years
    fire_info = fire_info.copy()
    fire_info["start_year"] = fire_info["start"].dt.year
    fire_info["end_year"] = fire_info["end"].dt.year
    start_years = set(pd.unique(fire_info["start_year"]))
    end_years = set(pd.unique(fire_info["end_year"]))
    years = start_years.union(end_years)
    years = [x for x in years if pd.notna(x)]
    years.sort()

    map_base = folium.Map(tiles=None, prefer_canvas=True)

    for year in years:
        fire_info_year = fire_info[(fire_info["start_year"] == year) |
                                   (fire_info["end_year"] == year)]
        feature_group = folium.FeatureGroup(str(year)).add_to(map_base)
        fire_info_year.apply(draw_circle, group=feature_group, spread_as_radius=True, axis=1)

    # max possible map zoom
    map_base.fit_bounds(map_base.get_bounds())

    folium.TileLayer("openstreetmap").add_to(map_base)
    folium.TileLayer("stamenterrain").add_to(map_base)
    folium.LayerControl(collapsed=False).add_to(map_base)

    return map_base


def make_info_map(station_info=None, fire_info=None):
    """
    Makes map to visualize station and fire locations.

    Map includes station location markers, fire location circles with spread, and lines connecting each fire with the
    station nearest to it.

    :param station_info: station info DataFrame or GeoJSON dictionary
    :param fire_info: fire info DataFrame
    :return: map or None if station_info and fire_info parameters are both None or empty
    """
    if utils.get_data_label(station_info) == "dict":
        station_info_dicts = []
        for station in station_info["features"]:
            station["properties"]["lat"] = station["geometry"]["coordinates"][1]
            station["properties"]["lon"] = station["geometry"]["coordinates"][0]
            station_info_dicts.append(station["properties"])
        station_info = pd.DataFrame(station_info_dicts)

    has_station = False
    if station_info is not None:
        if not station_info.empty:
            has_station = True
    has_fire = False
    if fire_info is not None:
        if not fire_info.empty:
            has_fire = True
    if not has_station and not has_fire:
        return None

    map_base = folium.Map(tiles=None, prefer_canvas=True)

    # add stations
    if has_station:
        feature_group = folium.FeatureGroup("Stations").add_to(map_base)
        station_info.apply(draw_circlemarker, group=feature_group, datetime_to_year=True, axis=1)

        # map zoom with stations
        #sw_bound = station_info[["lat", "lon"]].min().values.tolist()
        #ne_bound = station_info[["lat", "lon"]].max().values.tolist()
        #map_base.fit_bounds([sw_bound, ne_bound])

    # add fires
    if has_fire:
        feature_group = folium.FeatureGroup("Fires").add_to(map_base)
        fire_info.apply(draw_circle, group=feature_group, spread_as_radius=True, axis=1)

        # add connecting lines
        if has_station:
            # station distance for every fire point
            combined = fire_info[["lat", "lon"]].merge(station_info[["lat", "lon"]],
                                                       how="cross", suffixes=("_fire", "_station"))
            combined["distance"] = combined.apply(lambda x: math.sqrt((x[0] - x[2])**2 + (x[1] - x[3])**2), axis=1)

            # get nearest station point
            min_index = combined.groupby(["lat_fire", "lon_fire"])["distance"].idxmin()

            feature_group = folium.FeatureGroup("Nearest Stations").add_to(map_base)
            combined.loc[min_index].apply(draw_polyline, group=feature_group, suffixes=("_fire", "_station"), axis=1)
        #else:
            # max possible map zoom if only fires given
            #map_base.fit_bounds(map_base.get_bounds())

    map_base.fit_bounds(map_base.get_bounds())

    folium.TileLayer("openstreetmap").add_to(map_base)
    folium.TileLayer("stamenterrain").add_to(map_base)
    #folium.TileLayer("cartodbpositron").add_to(map)
    #folium.TileLayer("cartodbdark_matter").add_to(map)
    #folium.TileLayer("stamentoner").add_to(map)
    folium.LayerControl(collapsed=False).add_to(map_base)

    return map_base


def make_and_output_network_info_maps(start_datetime=None, end_datetime=None, networks=None, screenshot=False):
    """
    Makes network info maps with stations and fires and outputs them to files.

    :param start_datetime: start datetime for data as Timestamp object or None to use the minimum pandas Timestamp
    :param end_datetime: end datetime for data as Timestamp object or None to use the maximum pandas Timestamp
    :param networks: network code, network code list, or network info DataFrame to map each selected network in a
                     separate file, or None to map all available networks in one file
    :param screenshot: take a screenshot of every created map and output it to file if True (no output if process times
                       out)
    """
    item_type = utils.get_data_label(networks)
    if item_type == "df":
        networks = networks["network"].tolist()
    elif item_type == "str":
        networks = [networks]

    # input info
    station_info = utils.read_station_info(networks=networks, start_datetime=start_datetime, end_datetime=end_datetime)
    fire_info = utils.read_fire_info(networks=networks, start_datetime=start_datetime, end_datetime=end_datetime)

    if networks is not None:
        # map for every network separately
        for network in networks:
            if station_info is not None:
                station_info_network = station_info[station_info["network"].isin([network])]
            else:
                station_info_network = None

            if fire_info is not None:
                fire_info_network = fire_info[fire_info["network"].isin([network])]
            else:
                fire_info_network = None

            network_map = make_info_map(station_info_network, fire_info_network)

            # output info map
            if network_map is not None:
                file_out = utils.join_file(network, start_datetime, end_datetime, sep="_", ext=".html", convert=True)
                directory_out = utils.join_path("maps", network, root=True)
                utils.write_data_to_file(network_map, file_out, directory_out)

                # screenshot outputted map
                if screenshot:
                    file_path_in = utils.join_path(directory_out, file_out)  # html map
                    file_out = utils.join_file(network, start_datetime, end_datetime, sep="_", ext=".png", convert=True)

                    hti = Html2Image(output_path=directory_out,
                                     custom_flags=["--virtual-time-budget=1000", "--timeout=10000"])
                    hti.screenshot(html_file=file_path_in, save_as=file_out)

            network_map_fire = make_yearly_fire_info_map(fire_info_network)

            # output yearly fire info map
            if network_map_fire is not None:
                file_out = utils.join_file(network, start_datetime, end_datetime, "fires", sep="_", ext=".html",
                                           convert=True)
                directory_out = utils.join_path("maps", network, root=True)
                utils.write_data_to_file(network_map_fire, file_out, directory_out)

                # screenshot outputted map
                if screenshot:
                    file_path_in = utils.join_path(directory_out, file_out)  # html map
                    file_out = utils.join_file(network, start_datetime, end_datetime, "fires", sep="_", ext=".png",
                                               convert=True)

                    hti = Html2Image(output_path=directory_out,
                                     custom_flags=["--virtual-time-budget=1000", "--timeout=10000"])
                    hti.screenshot(html_file=file_path_in, save_as=file_out)
    else:
        # map for all available networks combined
        network_map = make_info_map(station_info, fire_info)

        # output info map
        if network_map is not None:
            file_out = utils.join_file("ALL", start_datetime, end_datetime, sep="_", ext=".html", convert=True)
            directory_out = utils.join_path("maps", "ALL", root=True)
            utils.write_data_to_file(network_map, file_out, directory_out)

            # screenshot outputted map
            if screenshot:
                file_path_in = utils.join_path(directory_out, file_out)  # html map
                file_out = utils.join_file("ALL", start_datetime, end_datetime, sep="_", ext=".png", convert=True)

                hti = Html2Image(output_path=directory_out,
                                 custom_flags=["--virtual-time-budget=1000", "--timeout=10000"])
                hti.screenshot(html_file=file_path_in, save_as=file_out)

        network_map_fire = make_yearly_fire_info_map(fire_info)

        # output yearly fire info map
        if network_map_fire is not None:
            file_out = utils.join_file("ALL", start_datetime, end_datetime, "fires", sep="_", ext=".html",
                                       convert=True)
            directory_out = utils.join_path("maps", "ALL", root=True)
            utils.write_data_to_file(network_map_fire, file_out, directory_out)

            # screenshot outputted map
            if screenshot:
                file_path_in = utils.join_path(directory_out, file_out)  # html map
                file_out = utils.join_file("ALL", start_datetime, end_datetime, "fires", sep="_", ext=".png",
                                           convert=True)

                hti = Html2Image(output_path=directory_out,
                                 custom_flags=["--virtual-time-budget=1000", "--timeout=10000"])
                hti.screenshot(html_file=file_path_in, save_as=file_out)
