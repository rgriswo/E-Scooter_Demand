#!/usr/bin/env python
# coding: utf-8

import csv, sys, os, glob, datetime, pickle, math
import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pylab as plt

# global constants
INTERVAL = 60 * 60       # a hour
OFFSET_STAMP = datetime.datetime(2016,7,7,4,0).timestamp()
# datapath="/home/dskim/winhome/Documents/bike-share/metro-bike-share"
DATAPATH = "C:\\Users\\dskim\\OneDrive - Indiana University\\workspace\\data\\e-scooter"
FILENAME = ["Microtransit__Scooter_and_Ebike__Trips.csv"]

# define functions
def time2tag(x):
    return int(dateconv(x).timestamp()-OFFSET_STAMP)//INTERVAL

def tag2time(x):
    return datetime.datetime.fromtimestamp(x * INTERVAL + OFFSET_STAMP)

def dateconv(x):
    if x.count('-') > 0:
        format = "%Y-%m-%d"
    else:
        format = "%m/%d/%Y"    
    if x.count(':') > 0:
        format += " %H:%M"
        if x.count(':') > 1:
            format += ":%S"
    dt = datetime.datetime.strptime(x, format)
    return dt


# to read station files 
def read_trip_files(filename): 
    if filename is list: 
        # station_filenames = sorted(glob.glob(filebase))
        df = pd.concat(map(pd.read_csv, filename))
        # df = df.groupby('id').first()
        # df.rename(columns={'date':'time'}, inplace=True) 
    else: 
        df = pd.read_csv(filename)
    return df

# To collect station info from trip files. 
# for both start_station and end_station
# Because each trip file has little different columns, 
# the file are read individually and then updated correspondingly
def collect_stations(filename):
    df = pd.read_csv(filename, low_memory=False)       
    if 'start_station_id' in df.columns: 
        df.rename(columns={'start_station_id': 'start_station', 
                           'end_station_id': "end_station"}, inplace=True) 
    s0 = df[['start_station','start_lat','start_lon','start_time']].groupby('start_station').first()
    s0.reset_index(inplace=True)
    s0.rename(columns={'start_station':'id', 'start_lat':"lat", 'start_lon':'lon', 'start_time':'time'}, inplace=True)
    s1 = df[['end_station', 'end_lat', 'end_lon', 'end_time']].groupby('end_station').first()
    s1.reset_index(inplace=True)
    s1.rename(columns={'end_station': 'id', 'end_lat':'lat', 'end_lon':'lon', 'end_time':'time'}, inplace=True)
    
    st = pd.concat([s0, s1]).groupby('id').first()
    st.reset_index(inplace=True)
    st = st.astype({'id': 'int32'})    
    del df
    del s0
    del s1
    return st




# In[9]:


def build_trip_db(filename, num_station): 
    # initialize variables
    odlist = []
    od = None
    ct = -1
    matrix_shape = (num_station, num_station)
    maxval = 0

    for fn in filename:
        print("reading %s" % os.path.basename(fn))
        df = pd.read_csv(fn, low_memory=False)     
        if 'start_station_id' in df.columns:     # some file has different column names
            df.rename(columns={'start_station_id': 'start_station', 'end_station_id': "end_station"}, inplace=True)   

        df['start_tag'] = df['start_time'].apply(time2tag)
        columns_needed = {'duration', 'start_station', 'end_station', 'start_tag'}
        drop_columns = set(df.columns).difference(columns_needed)
        df = df.drop(columns=drop_columns)
        df.sort_values('start_tag', inplace=True)

        # assume that data for one interval can be spread for multiple files.
        for k in range(len(df)):
            if df.iloc[k]['start_tag'] != ct and od != None: # a new tag start, but not the first data
                m = max(maxval, np.amax(xdata))
                if m != maxval: 
                    maxval = m
                    # print(maxval)
                    
                od['mat'] = sparse.coo_matrix(xdata)         # to stored the current work to the list
                odlist.append(od)

            if df.iloc[k]['start_tag'] != ct:                    # a new tag start 
                for tt in range(ct+1, df.iloc[k]['start_tag']):  # create empty slots to fill up the gap
                    od = {'tag':tt, 'dat':tag2time(tt), 'cnt':0,
                          'mat':sparse.coo_matrix(np.zeros(matrix_shape, dtype=np.uint8))}
                    odlist.append(od)

                ct = df.iloc[k]['start_tag']
                od = {'tag':ct, 'dat': tag2time(ct), 'cnt':0}
                xdata = np.zeros(matrix_shape, dtype=np.uint8)
            row = station.loc[df.iloc[k]['start_station'],'tidx']
            col = station.loc[df.iloc[k]['end_station'],'tidx']
            xdata[row][col] += 1
            od['cnt'] += 1

        del df       # free a large memory for possible run-time error

    # flush the pending od table    
    od['mat'] = sparse.csr_matrix(xdata)
    # print("append ", od['tag'])
    odlist.append(od)
    return odlist

def build_grid_db(filename):
    pass

# In[11]:
    
def draw_point(latitude, longitude, lat_limit, lon_limit):
    margin = 2*0.0005
    box = (lon_limit[0]-margin, lon_limit[1]+margin, lat_limit[0]-margin, lat_limit[1]+margin)
    rm = plt.imread('map-kansas.png')
    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(longitude, latitude, zorder=1, alpha=0.2, c='b', s=10)
    ax.set_title('Plotting Chicago')
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2],box[3])
    ax.imshow(rm, zorder=0, extent=box, aspect='equal')
    plt.show()   
     
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
    
DELTA = 0.0005
if __name__ == "__main__":
    print("building station_db")
    filename = os.path.join(DATAPATH, FILENAME[0])
    # station = build_grid_db(filename)
    
    g = pd.read_csv(filename)  #, nrows=2000)    
    # 2000 rows bounded at 
    # ((38.981, 39.1525), (-94.6225, -94.4735))
    
    # all rows bounded at 
    # ((0.0, 48.2115), (-121.506, 16.3745))
    latitude = np.concatenate((g['Start Latitude'],g['End Latitude']))    
    longitude = np.concatenate((g['Start Longitude'],g['End Longitude']))
    
    
    filt = np.isfinite(latitude) & np.isfinite(longitude)
    latitude = latitude[filt]
    longitude = longitude[filt]
    # filter outliers
    f0 = (longitude > -95) & (longitude < -94) & (latitude > 38) & (latitude < 40)
    longitude = longitude[f0]
    latitude = latitude[f0]
    bbox = BoundBox(latitude, longitude, DELTA)
    lat, lon = bbox.limits()    
    
    # g = pd.read_csv(filename)      
    # latitude = np.concatenate((g['Start Latitude'],g['End Latitude']))    
    # longitude = np.concatenate((g['Start Longitude'],g['End Longitude']))
    # filt = np.isfinite(latitude) & np.isfinite(longitude)
    # latitude = latitude[filt]
    # longitude = longitude[filt]   
    
    draw_point(latitude, longitude, lat, lon)
    
    

    # print("building trip_db")
    # filename = sorted(glob.glob(os.path.join(datapath, "*-trips-*.csv")))
    # odlist = build_trip_db(filename, len(station))

    # print("saving the db to file")
    # with open('data_station-v2.pkl', 'wb') as sout:
    #     pickle.dump(station, sout, pickle.HIGHEST_PROTOCOL)
        
    # with open('data_odpair-v2-coo.pkl', 'wb') as sout:
    #     pickle.dump(odlist, sout, pickle.HIGHEST_PROTOCOL)

