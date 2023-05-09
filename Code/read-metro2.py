#!/usr/bin/env python
# coding: utf-8

# In[5]:


import csv, sys, os, glob, datetime, pickle
import pandas as pd
import numpy as np
from scipy import sparse


# In[6]:


# global constants
INTERVAL = 60 * 60       # a hour
OFFSET_STAMP = datetime.datetime(2016,7,7,4,0).timestamp()
datapath="/home/dskim/winhome/Documents/bike-share/metro-bike-share"


# In[7]:


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


# datestr = ["7/7/2016", "10/9/2022 9:04", "2017-05-09 10:29:00"]
# [dateconv(x) for x in datestr]


# In[13]:


# to read station files 
def read_station_files(filebase): 
    station_filenames = sorted(glob.glob(filebase))
    df = pd.concat(map(pd.read_csv, station_filenames))
    df = df.groupby('id').first()
    df.rename(columns={'date':'time'}, inplace=True) 
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

def build_station_db(datapath):
    stabase = os.path.join(datapath, "metro-bike-share-stations-*.csv")
    station_supplement = read_station_files(stabase)
    tripbase = os.path.join(datapath, "*-trips-*.csv")
    tripname = sorted(glob.glob(tripbase))
    
    # cannot generally applies pd.concat(map(...)) for reading 
    # because column format are asimilar.
    station_per_trip = [collect_stations(x)  for x in tripname] 
    station_all = pd.concat(station_per_trip).groupby('id').first()
    
    sx = pd.concat([station_supplement, station_all]).groupby('id').first()
    sx['datetime'] = sx['time'].apply(dateconv)
    sx['tidx'] = range(len(sx))
    sx = sx.drop(columns=['time'])
    del station_per_trip
    del station_all
    return sx


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


# In[11]:

if __name__ == "__main__":
    print("building station_db")
    station = build_station_db(datapath)

    print("building trip_db")
    filename = sorted(glob.glob(os.path.join(datapath, "*-trips-*.csv")))
    odlist = build_trip_db(filename, len(station))

    # print("saving the db to file")
    # with open('data_station-v2.pkl', 'wb') as sout:
    #     pickle.dump(station, sout, pickle.HIGHEST_PROTOCOL)
        
    with open('data_odpair-v2-coo.pkl', 'wb') as sout:
        pickle.dump(odlist, sout, pickle.HIGHEST_PROTOCOL)

