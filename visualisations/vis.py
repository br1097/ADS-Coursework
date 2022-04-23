import datetime
import numpy as np 
import netCDF4 as nc
from scipy import stats
from lib import plot_time_series

PATH = "data/1991TS"
FILENAME = "fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.4p4km.nc"

fn = f"{PATH}/{FILENAME}"
fg = nc.Dataset(fn)
    
lat = fg.variables["latitude"][:]
lng = fg.variables["longitude"][:]
wind_speed = fg.variables["wind_speed_of_gust"][:]
print("Threshold started", datetime.datetime.now().utcnow().__str__())
thresh_wind = np.array(
    [[np.where(frame < wind_speed.max() * .3, np.nan, frame) for frame in ensemble ] for ensemble in wind_speed ])

below_wind = np.array(
    [[np.where(frame < wind_speed.max() * .3, frame, np.nan) for frame in ensemble ] for ensemble in wind_speed ])

clouds = np.array(
    [[np.where(stats.zscore(frame, nan_policy="omit") > -1, frame, np.nan) for frame in ensemble] for ensemble in below_wind])

print("Threshold completed", datetime.datetime.now().utcnow().__str__())

plot_time_series('contour.gif', lng, lat, np.array(thresh_wind), clouds=clouds)
# plot_time_series("contour.gif", lng, lat, np.array([wind_speed[0]]))