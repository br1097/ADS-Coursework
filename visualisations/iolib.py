import os
import re
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import imageio.v3 as iio
import numpy as np

from dotenv import load_dotenv
load_dotenv()


ensemble = 8


if os.getcwd()[-14:] == "ADS-Coursework":
    OUTPUT_DIRNAME = "visualisations/out"
else:
    OUTPUT_DIRNAME = "../visualisations/out"


cloud_map = ListedColormap([[1, 1, 1, n/4 if n > 0 else 0] for n in range(4)], "clouds")

matplotlib.rcParams["font.family"] = "Kohinoor Bangla"

cities_bangla = [
        {
        "name": "চট্টগ্রাম",
        "lat": 22.3569,
        "lng": 91.7832
    },
    {
        "name": "ঢাকা",
        "lat": 23.8103,
        "lng": 90.4125
    }]
cities_romanticised = [
        {
        "name": "Chattogram",
        "lat": 22.3569,
        "lng": 91.7832
    },
    {
        "name": "Dhaka",
        "lat": 23.8103,
        "lng": 90.4125
    }]

def plot_wind_speed(lng: np.array, lat: np.array, frame: np.array, levels: np.array, image=None, clouds: np.ndarray=None, cloud_levels: np.ndarray=None, size="4p4"):
    fig = plt.figure()
    fig.set_size_inches(20, 20)
    
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lng.min(), lng.max(), lat.min(), lat.max()])
    
    if image is None:
        print("WARNING: Downloading image in plot_wind_speed is innefficient, please specify image in time series")
        image = cimgt.MapboxTiles(os.environ.get("MAPBOX_TOKEN"), 'satellite-v9')
    ax.add_image(image, 6)

    fill_cntr = ax.contourf(lng, lat, frame, transform=ccrs.PlateCarree(), cmap="Wistia", levels=levels)
    if clouds is not None:
        cloud_cntr = ax.contourf(lng, lat, clouds, transform=ccrs.PlateCarree(), cmap=cloud_map, levels = cloud_levels)
        # fig.colorbar(cloud_cntr, ax=ax)
    # line_cntr = ax.contour(lng, lat, frame, levels=fill_cntr.levels, colors=['black'])
    fig.colorbar(fill_cntr, ax=ax, shrink=.7)
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS)
        
    plt.title("Wind Speed (m/s); Cyclone BoB 1; 1991", size=28)
    
    # Add City Tags
    for city in cities_bangla:
        plt.plot(city["lng"], city["lat"], "mo", markersize=12)
        delta = .2 if size == 4.4 else .1
        ax.annotate(city["name"], xy=(city["lng"] + delta, city["lat"] + delta/2), size=28, color="#D0DF")

    return fig, ax

def plot_time_series(output_filename: str, lng: np.ndarray, lat: np.ndarray, variable: np.ndarray, clouds: np.ndarray=None, centroids=None, size="4p4"):
    """
    output_filename: str # must be .gif
    lng: np.ndarray # longitude length 810
    lat: np.ndarray # latitude length 790
    variable: np.ndarray # the visualised variable of shape (reference_time, ensemble, lat, lng)\
    centroids: np.ndarray # cluster centres to plot, one for every frame atm
    """
    assert re.match(r'[^\/\\.]+\.mp4', output_filename)
    assert len(variable.shape) == 4
    filenames = []
    maxlevel = np.nanmax(variable)
    
    steps = 7
    levels = [((maxlevel)/steps) * val for val in list(range(steps + 1))]
    
    if clouds is not None:
        maxlevel = np.nanmax(clouds)
        steps = 7
        cloud_levels = [((maxlevel)/steps) * val for val in list(range(steps + 1))]
    else:
        cloud_levels = None
    
    assert os.environ.get("MAPBOX_TOKEN") is not None
    
    image = cimgt.MapboxTiles(os.environ.get("MAPBOX_TOKEN"), 'satellite-v9')
    
     
    
    for idx, frame in enumerate(variable):
        if frame[ensemble].shape != (lat.shape[0], lng.shape[0]):
            error = f"Frame shape incorrect, expected: {(len(lat), len(lng))} got: {frame.shape}"
            raise Exception(error)
        # make the plot, TODO change to kernel function 
        fig, ax = plot_wind_speed(lng, lat, frame[ensemble], levels, image, clouds[idx][ensemble] if clouds is not None else None, cloud_levels=cloud_levels, size=size)
        
        # Add centroids
        if centroids:
            ax.plot(centroids[idx][0], centroids[idx][1], "ro")
            
        # output file
        filename = f"{OUTPUT_DIRNAME}/sc{idx}.png"
        
        # Bottleneck
        filenames.append(filename)
        print(f"Before savefig: {idx}", datetime.datetime.now().utcnow().__str__())
        plt.savefig(filename)
        print(f"After savefig: {idx}", datetime.datetime.now().utcnow().__str__())
        plt.close()
        print(f"Done with frame: {idx}", datetime.datetime.now().utcnow().__str__())
                
    # build gif
    frames = np.stack(
                [iio.imread(filename) for filename in filenames],
                axis=0
            )
    
    iio.imwrite(f"{OUTPUT_DIRNAME}/{output_filename}", frames)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)