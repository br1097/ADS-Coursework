import datetime
import sys
import numpy as np 
import netCDF4 as nc
from scipy import stats
from iolib import plot_time_series, plot_centroid_path, plot_blank_map, plot_single_frame, plot_single_centroid
from memory import get_centroids
from mllib import z_score_threshold, mag_m_step
import pandas as pd
import matplotlib.pyplot as plt




if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "centroids":
            PATH = "data/1991TS"
            FILENAME = "fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.4p4km.nc"

            fn = f"{PATH}/{FILENAME}"
            fg = nc.Dataset(fn)

            lat = fg.variables["latitude"][:]
            lng = fg.variables["longitude"][:]
            centroids = get_centroids()
            
            filtered = centroids[list(filter(lambda x: x % 5 == 0, list(range(len(centroids)))))]
            plot_centroid_path('centroids4p4.png', lng, lat, filtered)
        
        elif sys.argv[1] == "wind_speed":
            PATH = "data/1991TS"
            FILENAME = "fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.4p4km.nc"

            fn = f"{PATH}/{FILENAME}"
            fg = nc.Dataset(fn)

            lat = fg.variables["latitude"][:]
            lng = fg.variables["longitude"][:]
            wind_speed = fg.variables["wind_speed_of_gust"][:]
            
            print("Threshold started", datetime.datetime.now().utcnow().__str__())
            thresh_wind, below_wind = z_score_threshold(wind_speed)

            clouds = np.array(
                [[np.where(stats.zscore(frame, nan_policy="omit") > -1, frame, np.nan) for frame in ensemble] for ensemble in below_wind])

            print("Threshold completed", datetime.datetime.now().utcnow().__str__())
            plot_time_series('wind_speed4p4.mp4', lng, lat, np.array(thresh_wind), clouds=clouds, size="4p4")
            
        elif sys.argv[1] == "report":
            PATH = "data/1991TS"
            # FILENAME = "fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.4p4km.nc"
            
            # fn = f"{PATH}/{FILENAME}"
            # fg = nc.Dataset(fn)
            
            # lat = fg.variables["latitude"][:]
            # lng = fg.variables["longitude"][:]
            # wind_speed4p4 = fg.variables["wind_speed_of_gust"][:]
            
            # print("Threshold started", datetime.datetime.now().utcnow().__str__())
            # thresh_wind, below_wind = z_score_threshold(wind_speed4p4)
            # print("Threshold ended", datetime.datetime.now().utcnow().__str__())

            # clouds = np.array(
            #     [[np.where(stats.zscore(frame, nan_policy="omit") > -1, frame, np.nan) for frame in ensemble] for ensemble in below_wind])
            
            # plot_single_frame("wind_speed_full_vis4p4.png", lng, lat, thresh_wind[0][0], clouds=clouds[0][0], size="4p4")
            
            
            FILENAME = "fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.1p5km.nc"
            
            fn = f"{PATH}/{FILENAME}"
            fg = nc.Dataset(fn)
            
            lat = fg.variables["latitude"][:]
            lng = fg.variables["longitude"][:]
            wind_speed1p5 = fg.variables["wind_speed_of_gust"][:]
            
            print("Threshold started", datetime.datetime.now().utcnow().__str__())
            thresh_wind, below_wind = z_score_threshold(wind_speed1p5)
            print("Threshold ended", datetime.datetime.now().utcnow().__str__())

            clouds = np.array(
                [[np.where(stats.zscore(frame, nan_policy="omit") > -1, frame, np.nan) for frame in ensemble] for ensemble in below_wind])
            
            plot_single_frame("wind_speed_full_vis1p5.png", lng, lat, thresh_wind[12][8], clouds=clouds[12][8], size="1p5")
            # plot_time_series("wind_speed_report.mp4", lng, lat, thresh_wind, clouds, size="1p5")
            
        elif sys.argv[1] == "pressure":
            
            PATH = "data/1991TS"
            FILENAME = "zg.T3Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc"

            fn = f"{PATH}/{FILENAME}"
            psl = nc.Dataset(fn)

            lat = psl.variables["latitude"][:]
            lng = psl.variables["longitude"][:]
            pressure = psl.variables["geopotential_height"][:]
            
            plot_time_series("pressure.mp4", lng, lat, pressure, kind="pressure")
        elif sys.argv[1] == "blank":
            PATH = "data/1991TS"
            FILENAME = "zg.T3Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc"

            fn = f"{PATH}/{FILENAME}"
            psl = nc.Dataset(fn)

            lat4p4 = psl.variables["latitude"][:]
            lng4p4 = psl.variables["longitude"][:]
            
            PATH = "data/1991TS"
            FILENAME = "zg.T3Hpoint.UMRA2T.19910428_19910501.BOB01.1p5km.nc"

            fn = f"{PATH}/{FILENAME}"
            psl = nc.Dataset(fn)

            lat1p5 = psl.variables["latitude"][:]
            lng1p5 = psl.variables["longitude"][:]
            
            
            plot_blank_map("blank4p4.png", lng4p4, lat4p4, "4.4")
            plot_blank_map("blank1p5.png", lng1p5, lat1p5, "1.5")
            
        elif sys.argv[1] == "report_clusters":
            PATH = "data/1991TS"
            FILENAME = "fg.T1Hmax.UMRA2T.19910428_19910501.BOB01.4p4km.nc"

            fn = f"{PATH}/{FILENAME}"
            fg = nc.Dataset(fn)

            lat = fg.variables["latitude"][:]
            lng = fg.variables["longitude"][:]
            wind_speed = fg.variables["wind_speed_of_gust"][:]

            thresh_wind, _ = z_score_threshold(np.array([[wind_speed[0][0]]]))
            frame = thresh_wind[0][0]
            centroid = mag_m_step(frame, lng, lat)
            plot_single_centroid("centroid.png", lng, lat, frame, centroid)
    else:
        print("Please input argument")



            