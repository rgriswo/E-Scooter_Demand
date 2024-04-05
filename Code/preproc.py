#!/usr/bin/env python
"""
Created on Sat May 13 18:37:53 2023
@author: Ryan and Stephen
"""
import sys
import pandas as pd
# import header_names as hd
import math as m
from datetime import datetime
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from scipy import stats
import numpy as np
import pickle
import torch
import pathlib
from matplotlib import pyplot as plt
from readconfig import read_config

###########global constants#################
EARTH_RADIUS = 6371         # in kilometer
EARTH_CIRCUM = EARTH_RADIUS * 2 * m.pi * 1000    # meterdf[]

#############################################
config = {}
   
def z_score(x,u,r):
    return (x-u)/r

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(m.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = m.sin(dlat/2)**2 + m.cos(lat1) * m.cos(lat2) * m.sin(dlon/2)**2
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1-a))
    base = c * EARTH_RADIUS
    return base

def find_bbox(df):
    lat = np.concatenate((df['start_lat'], df['end_lat'])) 
    lon = np.concatenate((df['start_lon'], df['end_lon']))
    min_lat = min(lat)
    max_lat = max(lat)
    min_lon = min(lon)
    max_lon = max(lon)
    # print("find_bbox()")
    # print("min, max =", min_lat, max_lat, min_lon, max_lon)    
    ALIGN_FACTOR = config['general']['align_factor']
    min_lat = m.floor(min_lat * ALIGN_FACTOR) / ALIGN_FACTOR
    max_lat = m.ceil (max_lat * ALIGN_FACTOR) / ALIGN_FACTOR
    min_lon = m.floor(min_lon * ALIGN_FACTOR) / ALIGN_FACTOR
    max_lon = m.ceil (max_lon * ALIGN_FACTOR) / ALIGN_FACTOR
    # print("min, max =", min_lat, max_lat, min_lon, max_lon)    
    # print("diff lat, lon =", max_lat - min_lat, max_lon - min_lon)
    return [[min_lat,max_lat],[min_lon,max_lon]]

def gps2grid(lat,lon, min_lat, min_lon):
    # lat and lon can be pandas series
    #calculate the distance
    dx = (lon-min_lon)*EARTH_CIRCUM*m.cos(min_lat*m.pi/360)/360
    dy = (lat-min_lat)*EARTH_CIRCUM/360
    # divide distance by grid size
    dx = dx / config['general']['cell_size']
    dy = dy / config['general']['cell_size']      
    return dx.astype('int32'), dy.astype('int32')
      
def timestamp_rollover(x):
    if "T24:00" in x:
        ts = pd.Timestamp(x.replace("T24:00", "T00:00"))
        ts += pd.to_timedelta(1, unit='d')
    else:
        ts = pd.Timestamp(x)   
    return ts
# Open database have have different schema.  
#  
def formalize(df, kind):
    if kind == "TYPE_PURR":
        df['start_tag'] = df['start_time_utc'].apply(lambda x: pd.Timestamp(x))
        df['end_tag']   = df['end_time_utc'].apply(lambda x: pd.Timestamp(x))
    elif kind == "TYPE_LOUISVILLE":
        df['start_tag'] = df[['StartDate', 'StartTime']].agg('T'.join, axis=1).apply(lambda x: timestamp_rollover(x))
        df['end_tag'] = df[['EndDate', 'EndTime']].agg('T'.join, axis=1).apply(lambda x: timestamp_rollover(x))
        df = df.rename(columns={"StartLatitude":"start_lat", "StartLongitude":"start_lon",
                           "EndLatitude":"end_lat", "EndLongitude":"end_lon"
                           })
    elif kind == "TYPE_KANSAS":
        df['Start Date'] = df['Start Date'].apply(lambda x: x[:10])
        df['End Date'] = df['End Date'].apply(lambda x: x[:10])
        df['start_tag'] = df[['Start Date', 'Start Time']].agg('T'.join, axis=1).apply(lambda x: pd.Timestamp(x))
        df['end_tag'] = df[['End Date', 'End Time']].agg('T'.join, axis=1).apply(lambda x: pd.Timestamp(x))
        df = df.rename(columns={"Start Latitude":"start_lat", "Start Longitude":"start_lon",
                           "End Latitude":"end_lat", "End Longitude":"end_lon"
                           })
    else: 
        print("TYPE_UNKNOWN")
        
    return df[['start_tag', 'end_tag', 'start_lat', 'start_lon', 'end_lat', 'end_lon']]

#  to convert the start times in hour resolution to indices (time_slot)  
def calculate_time_index(df):
    m = min(df['start_tag'])
    r = max(df['end_tag'])
    print("date range =", m, r)
    offset = pd.Timestamp(year=m.year, month=m.month, day=m.day, hour=m.hour, tz=m.tzname())
    print("offset = ", offset)
    delta = df['start_tag'] - offset
    hour = delta.apply(lambda x: x.seconds/3600)
    df['time_slot'] = hour.astype(int)
    return df, offset


def remove_short_distance(df, threshold_distance):
    df['distance'] = df.apply(lambda x: haversine(x['start_lat'], x['start_lon'], x['end_lat'], x['end_lon']),
                              axis=1)    
    # remove rows whose travel distance is shorter than the minimum threshold
    df = df[ df['distance'] >= threshold_distance ]
    return df
    
def filter_outliers(df, zval=5.0):           
    # remove outliers with z scores   
    column_headers = [['start_lon', 'end_lon'],['start_lat', 'end_lat']] 
                            
    for coord_col in column_headers:
        #0 = start 1 = end
        #combine that start and end collums for either lat or lon
        coord_combined = np.concatenate((df[coord_col[0]], df[coord_col[1]]),axis = 0)
        for col in coord_col:
            #calculate the z_score for column using the population mean and std
            z_scores = z_score(df[col],coord_combined.mean(),coord_combined.std())
            df = df[(abs(z_scores) < zval)]
          
    return df


def database_schema(df):
    if 'start_time_utc' in df.columns:
        kind = 'TYPE_PURR'
    elif 'StartDate' in df.columns and 'StartTime' in df.columns:
        kind = 'TYPE_LOUISVILLE'
    elif 'Start Date' in df.columns and 'Start Time' in df.columns:
        kind = 'TYPE_KANSAS'
    else: 
        kind = 'TYPE_DEFAULT'
    return kind


def build_grid_database(df, bb):
    min_lat = bb[0][0]
    min_lon = bb[1][0]    
    df['start_x'], df['start_y'] = gps2grid(df['start_lat'], df['start_lon'],
                                            min_lat, min_lon)
    df['end_x'], df['end_y'] = gps2grid(df['end_lat'], df['end_lon'], 
                                        min_lat, min_lon)
    return df              
   
def create_trip_db(df):
    total_demand = []
    slot_list = []
    odlist = []
    dimx = 1 + max(np.concatenate([df['start_x'],df['end_x']]))
    dimy = 1 + max(np.concatenate([df['start_y'],df['end_y']]))
    
    df.sort_values("time_slot", inplace=True)
    
    curr_slot = -1          # initialize.  slot never be nagative
    origin = np.zeros((dimx,dimy), dtype=int)
    destin = np.zeros((dimx,dimy), dtype=int)     
            
    for i, r in df.iterrows():
        
        if curr_slot != r['time_slot']:
            if curr_slot >= 0:
                odlist.append([origin, destin, curr_slot])
                total_demand.append(origin.sum())
                slot_list.append(curr_slot)
                
            curr_slot = r['time_slot']
            origin = np.zeros((dimx,dimy), dtype=int)
            destin = np.zeros((dimx,dimy), dtype=int)         
            
        origin[ r['start_x'], r['start_y']] += 1
        destin[ r['end_x'],   r['end_y']]   += 1
        
    odlist.append([origin, destin, curr_slot])
    total_demand.append(origin.sum())
    slot_list.append(curr_slot)
    
    plt.figure()
    plt.title('Total Demand Graph')
    plt.plot(total_demand)  
    plt.scatter(slot_list, total_demand)
#    plt.savefig('Total_Demand_Graph.pdf')
    
    return odlist

def main(filename):  
    print("reading %s" % filename)
    df = pd.read_csv(filename, sep=',')
    print("Shape after reading \t\t\t\t", df.shape)
         
    # sanitize the data
    # remove null values
    df.dropna()
    # print("Shape after dropna \t\t", df.shape)
    
    # to convert data & time to Pandas Timestamp type
    # to rename columns to short underscored
    # to select only columns useful    
    db_type = database_schema(df)

    df = formalize(df, db_type) 
    print("Shape after formalize reading\t\t", df.shape)
    
    df = remove_short_distance(df, 0.01)    # remove trips less than 0.01km move  
    print("Shape after remove short     \t\t", df.shape)
    
    df = filter_outliers(df, zval=3)    
    print("Shape after filter outliers  \t\t", df.shape)
    
    df, offset = calculate_time_index(df)
    print("Shape after calculate time index \t", df.shape)

    #get list of all lat and longs from start and end groups
    print("start building grid database")
    bbox = find_bbox(df)            # [[min_lat, max_lat], [min_lon, max_lon]]
    
    df = build_grid_database(df, bbox)
    
    print("finished grid database")
      
    trip_db = create_trip_db(df)
    
    print("Created trib db")
    
    df.to_csv("out.csv", index = False)
    
    pkl_name = pathlib.PurePath.joinpath(pathlib.Path(filename).parent, pathlib.Path(filename).stem+'.pkl')
    
    #
    # todo: to save offset and bbox into the same pickle file
    #
    with open(pkl_name, 'wb') as sout:
        pickle.dump(trip_db, sout, pickle.HIGHEST_PROTOCOL)
    
    return df

if __name__ == "__main__":
    config = read_config('escooter.ini')
    df = main('purr.csv')
    # df = main('louisville.csv')
    # df = main('kansas.csv')
    # df = main(r'C:\Users\dskim\Documents\Data\scooter\purr_scooter_data.csv')
    # df = main(r'C:\Users\dskim\Documents\Data\scooter\lousiville-escooter-2018-2019.csv')
    # df = main(r'C:\Users\dskim\Documents\Data\scooter\Kansas-Microtransit__Scooter_and_Ebike__Trips.csv')
