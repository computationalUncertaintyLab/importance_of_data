#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature



if __name__ == "__main__":
    # Define city locations
    # Largest 3 cities for each U.S. state with coordinates
    locations = {
        "Alabama": [
            ("Birmingham", (33.5186, -86.8104)),
            ("Montgomery", (32.3668, -86.3000)),
            ("Huntsville", (34.7304, -86.5861)),
        ],
        "Alaska": [
            ("Anchorage", (61.2181, -149.9003)),
            ("Fairbanks", (64.8378, -147.7164)),
            ("Juneau", (58.3019, -134.4197)),
        ],
        "Arizona": [
            ("Phoenix", (33.4484, -112.0740)),
            ("Tucson", (32.2226, -110.9747)),
            ("Mesa", (33.4152, -111.8315)),
        ],
        "Arkansas": [
            ("Little Rock", (34.7465, -92.2896)),
            ("Fort Smith", (35.3859, -94.3985)),
            ("Fayetteville", (36.0822, -94.1719)),
        ],
        "California": [
            ("Los Angeles", (34.0522, -118.2437)),
            ("San Diego", (32.7157, -117.1611)),
            ("San Jose", (37.3382, -121.8863)),
        ],
        "Colorado": [
            ("Denver", (39.7392, -104.9903)),
            ("Colorado Springs", (38.8339, -104.8214)),
            ("Aurora", (39.7294, -104.8319)),
        ],
        "Connecticut": [
            ("Bridgeport", (41.1792, -73.1894)),
            ("New Haven", (41.3083, -72.9279)),
            ("Stamford", (41.0534, -73.5387)),
        ],
        "Delaware": [
            ("Wilmington", (39.7459, -75.5466)),
            ("Dover", (39.1582, -75.5244)),
            ("Newark", (39.6837, -75.7497)),
        ],
        "Florida": [
            ("Jacksonville", (30.3322, -81.6557)),
            ("Miami", (25.7617, -80.1918)),
            ("Tampa", (27.9506, -82.4572)),
        ],
        "Georgia": [
            ("Atlanta", (33.7490, -84.3880)),
            ("Augusta", (33.4735, -82.0105)),
            ("Columbus", (32.4608, -84.9877)),
        ],
        "Hawaii": [
            ("Honolulu", (21.3069, -157.8583)),
            ("Pearl City", (21.3972, -157.9752)),
            ("Hilo", (19.7074, -155.0859)),
        ],
        "Idaho": [
            ("Boise", (43.6150, -116.2023)),
            ("Meridian", (43.6121, -116.3915)),
            ("Nampa", (43.5407, -116.5635)),
        ],
        "Illinois": [
            ("Chicago", (41.8781, -87.6298)),
            ("Aurora", (41.7606, -88.3201)),
            ("Naperville", (41.7508, -88.1535)),
        ],
        "Indiana": [
            ("Indianapolis", (39.7684, -86.1581)),
            ("Fort Wayne", (41.0793, -85.1394)),
            ("Evansville", (37.9716, -87.5711)),
        ],
        "Iowa": [
            ("Des Moines", (41.5868, -93.6250)),
            ("Cedar Rapids", (42.0083, -91.6441)),
            ("Davenport", (41.5236, -90.5776)),
        ],
        "Kansas": [
            ("Wichita", (37.6872, -97.3301)),
            ("Overland Park", (38.9822, -94.6708)),
            ("Kansas City", (39.1142, -94.6275)),
        ],
        "Kentucky": [
            ("Louisville", (38.2527, -85.7585)),
            ("Lexington", (38.0406, -84.5037)),
            ("Bowling Green", (36.9685, -86.4808)),
        ],
        "Louisiana": [
            ("New Orleans", (29.9511, -90.0715)),
            ("Baton Rouge", (30.4515, -91.1871)),
            ("Shreveport", (32.5252, -93.7502)),
        ],
        "Maine": [
            ("Portland", (43.6615, -70.2553)),
            ("Lewiston", (44.1003, -70.2148)),
            ("Bangor", (44.8012, -68.7778)),
        ],
        "Maryland": [
            ("Baltimore", (39.2904, -76.6122)),
            ("Frederick", (39.4143, -77.4105)),
            ("Rockville", (39.0839, -77.1528)),
        ],
        "Massachusetts": [
            ("Boston", (42.3601, -71.0589)),
            ("Worcester", (42.2626, -71.8023)),
            ("Springfield", (42.1015, -72.5898)),
        ],
        "Michigan": [
            ("Detroit", (42.3314, -83.0458)),
            ("Grand Rapids", (42.9634, -85.6681)),
            ("Warren", (42.5145, -83.0147)),
        ],
        "Minnesota": [
            ("Minneapolis", (44.9778, -93.2650)),
            ("Saint Paul", (44.9537, -93.0900)),
            ("Rochester", (44.0121, -92.4802)),
        ],
        "Mississippi": [
            ("Jackson", (32.2988, -90.1848)),
            ("Gulfport", (30.3674, -89.0928)),
            ("Southaven", (34.9889, -90.0126)),
        ],
        "Missouri": [
            ("Kansas City", (39.0997, -94.5786)),
            ("Saint Louis", (38.6270, -90.1994)),
            ("Springfield", (37.2089, -93.2923)),
        ],
        "Montana": [
            ("Billings", (45.7833, -108.5007)),
            ("Missoula", (46.8721, -113.9940)),
            ("Great Falls", (47.5052, -111.3008)),
        ],
        "Nebraska": [
            ("Omaha", (41.2565, -95.9345)),
            ("Lincoln", (40.8136, -96.7026)),
            ("Bellevue", (41.1544, -95.9146)),
        ],
        "Nevada": [
            ("Las Vegas", (36.1699, -115.1398)),
            ("Henderson", (36.0395, -114.9817)),
            ("Reno", (39.5296, -119.8138)),
        ],
        "New Hampshire": [
            ("Manchester", (42.9956, -71.4548)),
            ("Nashua", (42.7654, -71.4676)),
            ("Concord", (43.2081, -71.5376)),
        ],
        "New Jersey": [
            ("Newark", (40.7357, -74.1724)),
            ("Jersey City", (40.7178, -74.0431)),
            ("Paterson", (40.9168, -74.1718)),
        ],
        "New Mexico": [
            ("Albuquerque", (35.0844, -106.6504)),
            ("Las Cruces", (32.3199, -106.7637)),
            ("Rio Rancho", (35.2334, -106.6645)),
        ],
        "New York": [
            ("New York City", (40.7128, -74.0060)),
            ("Buffalo", (42.8864, -78.8784)),
            ("Rochester", (43.1566, -77.6088)),
        ],
        "North Carolina": [
            ("Charlotte", (35.2271, -80.8431)),
            ("Raleigh", (35.7796, -78.6382)),
            ("Greensboro", (36.0726, -79.7910)),
        ],
        "North Dakota": [
            ("Fargo", (46.8772, -96.7898)),
            ("Bismarck", (46.8083, -100.7837)),
            ("Grand Forks", (47.9253, -97.0329)),
        ],
        "Ohio": [
            ("Columbus", (39.9612, -82.9988)),
            ("Cleveland", (41.4993, -81.6944)),
            ("Cincinnati", (39.1031, -84.5120)),
        ],
        "Oklahoma": [
            ("Oklahoma City", (35.4676, -97.5164)),
            ("Tulsa", (36.1539, -95.9928)),
            ("Norman", (35.2226, -97.4395)),
        ],
        "Oregon": [
            ("Portland", (45.5051, -122.6750)),
            ("Eugene", (44.0521, -123.0868)),
            ("Salem", (44.9429, -123.0351)),
        ],
        "Pennsylvania": [
            ("Philadelphia", (39.9526, -75.1652)),
            ("Pittsburgh", (40.4406, -79.9959)),
            ("Allentown", (40.6084, -75.4902)),
        ],
        "Rhode Island": [
            ("Providence", (41.8240, -71.4128)),
            ("Cranston", (41.7798, -71.4373)),
            ("Warwick", (41.7001, -71.4162)),
        ],
        "South Carolina": [
            ("Columbia", (34.0007, -81.0348)),
            ("Charleston", (32.7765, -79.9311)),
            ("North Charleston", (32.8546, -79.9748)),
        ],
        "South Dakota": [
            ("Sioux Falls", (43.5473, -96.7283)),
            ("Rapid City", (44.0805, -103.2310)),
            ("Aberdeen", (45.4647, -98.4865)),
        ],
        "Tennessee": [
            ("Nashville", (36.1627, -86.7816)),
            ("Memphis", (35.1495, -90.0490)),
            ("Knoxville", (35.9606, -83.9207)),
        ],
        "Texas": [
            ("Houston", (29.7604, -95.3698)),
            ("San Antonio", (29.4241, -98.4936)),
            ("Dallas", (32.7767, -96.7970)),
        ],
        "Utah": [
            ("Salt Lake City", (40.7608, -111.8910)),
            ("West Valley City", (40.6916, -112.0010)),
            ("Provo", (40.2338, -111.6585)),
        ],
        "Vermont": [
            ("Burlington", (44.4759, -73.2121)),
            ("South Burlington", (44.4665, -73.1712)),
            ("Rutland", (43.6106, -72.9726)),
        ],
        "Virginia": [
            ("Virginia Beach", (36.8529, -75.9780)),
            ("Norfolk", (36.8508, -76.2859)),
            ("Chesapeake", (36.7682, -76.2875)),
        ],
        "Washington": [
            ("Seattle", (47.6062, -122.3321)),
            ("Spokane", (47.6588, -117.4260)),
            ("Tacoma", (47.2529, -122.4443)),
        ],
        "West Virginia": [
            ("Charleston", (38.3498, -81.6326)),
            ("Huntington", (38.4192, -82.4452)),
            ("Morgantown", (39.6295, -79.9559)),
        ],
        "Wisconsin": [
            ("Milwaukee", (43.0389, -87.9065)),
            ("Madison", (43.0731, -89.4012)),
            ("Green Bay", (44.5192, -88.0198)),
        ],
        "Wyoming": [
            ("Cheyenne", (41.1400, -104.8202)),
            ("Casper", (42.8501, -106.3252)),
            ("Laramie", (41.3114, -105.5911)),
        ]}

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-130, -60, 24, 50])  # Continental US

    # Add map features
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)

    # Plot city locations
    for state, cities in locations.items():
        for city, coords in cities:
            ax.scatter(coords[1], coords[0], color="red", s=10, alpha=0.7, transform=ccrs.PlateCarree())
            #ax.text(coords[1], coords[0], city, fontsize=6, ha="right", alpha=0.7, transform=ccrs.PlateCarree())

    ax.set_title("NOAA data measured at 3 largest cities per state", fontsize=14)

    fig.set_size_inches(8.5-2,(11-2)/3)
    
    plt.savefig("./viz/map_data.pdf")
    plt.close()
