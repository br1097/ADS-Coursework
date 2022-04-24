import profile
from mllib import mag_m_step
import netCDF4 as nc
import numpy as np

PATH = "data/1991TS"
# WIND SPEED OF GUST
FG = "fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.4p4km.nc"

fg = nc.Dataset(f"{PATH}/{FG}")

lat = fg.variables["latitude"][:]
lng = fg.variables["longitude"][:]
wind_speed = fg.variables["wind_speed_of_gust"][:][0]

# a single frame
var_frame = wind_speed[0]
threshold = np.where(var_frame < wind_speed.max() * .3, 0, var_frame)

profile.run("mag_m_step(threshold, lng, lat)")
