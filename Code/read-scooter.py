#!/usr/bin/env python
# coding: utf-8

import os, datetime    # pickle, math, csv, sys, glob
import pandas as pd
import numpy as np
# from scipy import sparse
import matplotlib.pylab as plt

# global constants
INTERVAL = 60 * 60       # a hour
DELTA = 0.0005
MARGIN = 2*DELTA
METRO_RADIUS = .3        # degree in latitude and longitude
OFFSET_STAMP = datetime.datetime(2016,7,7,4,0).timestamp()
# datapath="/home/dskim/winhome/Documents/bike-share/metro-bike-share"
DATAPATH = "C:\\Users\\dskim\\OneDrive - Indiana University\\workspace\\data\\e-scooter"
METROS = {'kansas': {'data': "Kansas-Microtransit__Scooter_and_Ebike__Trips.csv", 
                     'map':  "Kansas-map2.png",
                     'bbox': (-94.8560, -94.3000, 38.8195, 39.3800)},
          'norfolk': {'data': "Norfolk-Micromobility__Electric_Scooters_and_Bikes_.csv",
                      'map':  "Kansas-map2.png",
                      'bbox': (-94.8560, -94.3000, 38.8)}}

CITY  = 'kansas'
     
class BoundBox():
    def __init__(self, lat, lon, delta=0.001):
        self.delta = delta
        self.lon_min = np.floor(np.min(lon) / delta) * delta
        self.lat_min = np.floor(np.min(lat) / delta) * delta
        self.lon_max = np.ceil(np.max(lon) / delta) * delta
        self.lat_max = np.ceil(np.max(lat) / delta) * delta
        
    def lat_limit(self):
        return self.lat_min, self.lat_max
    
    def lon_limit(self):
        return self.lon_min, self.lon_max
    
    def limits(self):
        return self.lat_limit(), self.lon_limit()
    
    def shape(self):
        xcnt = int((self.lon_max-self.lon_min)/self.delta)
        ycnt = int((self.lat_max-self.lat_min)/self.delta)
        return xcnt, ycnt 
    
def draw_point(latitude, longitude, box=None):
    if box == None: 
        bbox = BoundBox(latitude, longitude, DELTA)
        lat, lon = bbox.limits()    
        box = (lon[0]-MARGIN, lon[1]+MARGIN, lat[0]-MARGIN, lat[1]+MARGIN)
        print(box)
        
    rm = plt.imread(METROS[CITY]['map'])
    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(longitude, latitude, zorder=1, alpha=0.2, c='b', s=10)
    ax.set_title('Plotting Chicago')
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.imshow(rm, zorder=0, extent=box, aspect='equal')
    # ax.imshow(rm, zorder=0, aspect='equal')
    plt.show()

# at latitude 42, longitude 1md is equavalent to 82.634m
# at latitude 38, longitude 1md is equavalent to 87.623m
# google map platform API Key: # AIzaSyDhJ9G0PWMmbmyXMdLv3rKJnWK6XlOQItI
# secrete generator:     -Hczp9OjijbaF9_KxEiE95DuoWY=
if __name__ == "__main__":
    print("building grid table")
    filename = os.path.join(DATAPATH, METROS[CITY]['data'])
    # station = build_grid_db(filename)
    
    
    g = pd.read_csv(filename)   
    # g = pd.read_csv(filename, nrows=2000)    
    
    lat = np.concatenate((g['Start Latitude'],g['End Latitude']))    
    lon = np.concatenate((g['Start Longitude'],g['End Longitude']))
    
    # filter out invalid GPS data
    filt_valid = np.isfinite(lat) & np.isfinite(lon)
    lat = lat[filt_valid]
    lon = lon[filt_valid]
    
    # filter out outliers
    # how to determine outliers 
    # about 30 km distance from the center  ????
    latc = lat.mean()
    lonc = lon.mean()    
    filt_range = (lonc-METRO_RADIUS < lon) & (lon < lonc+METRO_RADIUS) & (latc-METRO_RADIUS < lat) & (lat < latc+METRO_RADIUS)
    lat = lat[filt_range]
    lon = lon[filt_range]
    
    draw_point(lat, lon, METROS[CITY]['bbox'])
    