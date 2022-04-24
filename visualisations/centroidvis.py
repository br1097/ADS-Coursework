from datetime import datetime

from iolib import plot_time_series
from mllib import mag_m_step

import netCDF4 as nc
import numpy as np

PATH = "data/1991TS"
# WIND SPEED OF GUST
FG = "fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.4p4km.nc"

fg = nc.Dataset(f"{PATH}/{FG}")

lat = fg.variables["latitude"][:]
lng = fg.variables["longitude"][:]
wind_speed = fg.variables["wind_speed_of_gust"][:]

thresh_wind = np.array(
    [[np.where(frame < wind_speed.max() * .3, np.nan, frame) for frame in ensemble ] for ensemble in wind_speed ])

centroids = []
for idx, frame in enumerate(thresh_wind):
    print(f"Starting centroid {idx} \n At time: {datetime.now().utcnow().__str__()}")
    centroids.append(mag_m_step(frame, lng, lat))
    

with open('visualisations/out/centroids4p4.txt', 'w') as f:
    for item in centroids:
        f.write("%s\n" % item)
        
print(f"Done at time: {datetime.now().utcnow().__str__()} \nHere are the centroids: {centroids}")
    