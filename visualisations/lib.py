import os
import re
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import imageio
import numpy as np


OUTPUT_DIRNAME = "../visualisations/out"

cloud_map = ListedColormap([[1, 1, 1, n/4 if n > 0 else 0] for n in range(4)], "clouds")

matplotlib.rcParams["font.family"] = "Kohinoor Bangla"

cities = [
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

def plot_wind_speed(lng: np.array, lat: np.array, frame: np.array, levels: np.array, image=None, clouds: np.ndarray=None, cloud_levels: np.ndarray=None):
    fig = plt.figure()
    fig.set_size_inches(20, 20)
    
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lng.min(), lng.max(), lat.min(), lat.max()])
    
    if image is None:
        image = cimgt.MapboxTiles("pk.eyJ1IjoiYnIxOTA5NyIsImEiOiJjbDJhbXY0ZnIwNnlxM2JydWxtMXpmNHd3In0.ChkOob-if8f_diISrmJY2A", 'satellite-v9')
    ax.add_image(image, 6)

    fill_cntr = ax.contourf(lng, lat, frame, transform=ccrs.PlateCarree(), cmap="Wistia", levels=levels)
    if clouds is not None:
        cloud_cntr = ax.contourf(lng, lat, clouds, transform=ccrs.PlateCarree(), cmap=cloud_map, levels = cloud_levels)
        fig.colorbar(cloud_cntr, ax=ax)
    # line_cntr = ax.contour(lng, lat, frame, levels=fill_cntr.levels, colors=['black'])
    fig.colorbar(fill_cntr, ax=ax)
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS)
    
    # add cities
    for city in cities:
        ax.plot(city["lng"], city["lat"], color="black")
        ax.annotate(city["name"], xy=(city["lng"], city["lat"], ))

    return fig, ax
    
def plot_time_series(output_filename: str, lng: np.ndarray, lat: np.ndarray, variable: np.ndarray, clouds: np.ndarray=None, centroids=None):
    """
    output_filename: str # must be .gif
    lng: np.ndarray # longitude length 810
    lat: np.ndarray # latitude length 790
    variable: np.ndarray # the visualised variable of shape (reference_time, ensemble, lat, lng)\
    centroids: np.ndarray # cluster centres to plot, one for every frame atm
    """
    assert re.match(r'[^\/\\.]+\.gif', output_filename)
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
    
    image = cimgt.MapboxTiles("pk.eyJ1IjoiYnIxOTA5NyIsImEiOiJjbDJhbXY0ZnIwNnlxM2JydWxtMXpmNHd3In0.ChkOob-if8f_diISrmJY2A", 'satellite-v9')
    
    
    for idx, frame in enumerate(variable):
        fig, ax = plot_wind_speed(lng, lat, frame[8], levels, image, clouds[idx][8] if clouds is not None else None, cloud_levels=cloud_levels)
        if centroids:
            ax.plot(centroids[idx][0], centroids[idx][1], "ro")
        filename = f"{OUTPUT_DIRNAME}/sc{idx}.png"
        filenames.append(filename)
        print(f"Before savefig: {idx}", datetime.datetime.now().utcnow().__str__())
        plt.savefig(filename)
        print(f"After savefig: {idx}", datetime.datetime.now().utcnow().__str__())
        plt.close()
        print(f"Done with frame: {idx}", datetime.datetime.now().utcnow().__str__())
                
    # build gif
    with imageio.get_writer(f"{OUTPUT_DIRNAME}/{output_filename}", mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)