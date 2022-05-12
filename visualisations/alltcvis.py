from iolib import plot_all_paths
from memory import get_all_centroids
import netCDF4 as nc

centroids = get_all_centroids()

names = ["AILA", "BOB1", "AKASH",
                "BOB7",
                "FANI",
                "MORA",
                "RASHMI",
                "ROANU",
                "SIDR",
                "TC01B",
                "VIYARU",]

PATH = "data/1991TS"
FILENAME = "fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.4p4km.nc"

fn = f"{PATH}/{FILENAME}"
fg = nc.Dataset(fn)

lat = fg.variables["latitude"][:]
lng = fg.variables["longitude"][:]

filtered = [tc[list(filter(lambda x: x % 5 == 0, list(range(len(tc)))))] for tc in centroids]

filtered[2] = filtered[2][:-2]
filtered[3] = filtered[3][:-3]
filtered[4] = filtered[4][:-5]
plot_all_paths('allpaths4p4.png', lng, lat, filtered, names)