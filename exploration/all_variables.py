import sys
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt


BASE = "./data/1991TS"
OUTPUT_DIR = "exploration/vis"

files = ["fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "hur.T1Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "prlssn.T1Hmean.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "psl.T1Hmean.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "rsds.T1Hmean.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "rsnds.T1Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "tas.T1Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "ua.T1Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "va.T1Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "wbpt.T3Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "zg.T3Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc",
         "prlst.T1Hmean.UMRA2T.19910428_19910501.BOB01.4p4km.nc"]


ds = nc.Dataset(f"{BASE}/zg.T3Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc")

lat = ds.variables["latitude"][:]
long = ds.variables["longitude"][:]
geo_height = ds.variables["geopotential_height"][:][0][0]

rainfall = nc.Dataset(f"{BASE}/prlst.T1Hmean.UMRA2T.19910428_19910501.BOB01.4p4km.nc")["stratiform_rainfall_amount"][:]
wind_speed = nc.Dataset(f"{BASE}/fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.4p4km.nc")["wind_speed_of_gust"][:]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "vis":
            

            fig, ax = plt.subplots()
            colour = geo_height[4]

            cntr = ax.contourf(long, lat, colour)

            fig.colorbar(cntr, ax=ax)
            plt.margins(x=0, y=0)
            OUTPUT_DIR = "report_images"
            
            plt.title("Geopotential Height (m); Cyclone BoB 1; 1991; 4.4 km", size=12)
            plt.savefig(f"{OUTPUT_DIR}/geopotential_height.png", bbox_inches='tight')

            fig, ax = plt.subplots()
            cntr = plt.contourf(long, lat, rainfall[0][0])
            fig.colorbar(cntr, ax=ax)
            plt.title("Rainfall (kg/m^2); Cyclone BoB 1; 1991; 4.4 km", size=12)
            plt.savefig(f"{OUTPUT_DIR}/rainfall.png", bbox_inches='tight')
            
            fig, ax = plt.subplots()
            cntr = plt.contourf(long, lat, wind_speed[0][0])
            fig.colorbar(cntr, ax=ax)
            plt.title("Wind Speed (m/s); Cyclone BoB 1; 1991; 4.4 km", size=12)
            plt.savefig(f"{OUTPUT_DIR}/wind_speed.png", bbox_inches='tight')
        elif sys.argv[1] == "hist":
            plt.hist(rainfall[0][0].flatten(), bins=20)
            plt.savefig(f"{OUTPUT_DIR}/rainfall_hist.png", bbox_inches='tight')
            plt.close()
            
            plt.hist(wind_speed[0][0].flatten(), bins=20)
            plt.savefig(f"{OUTPUT_DIR}/wind_speed_hist.png", bbox_inches='tight')
            
        elif sys.argv[1] == "test":
            print(rainfall[0][0].shape, wind_speed[0][0].shape)
            
        elif sys.argv[1] == "var":
            print(f"Rainfall Variance: {np.var(rainfall[0][0].flatten())} \nWind Speed Variance: {np.var(wind_speed[0][0].flatten())}")
            
        else:
            print("Use options: \n    `vis` for visualisations\n    `hist` for histograms")
