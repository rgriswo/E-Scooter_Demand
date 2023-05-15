# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:37:53 2023

@author: Ryan's computer
"""

import pandas as pd
import header_names as hd
import math 
from datetime import datetime
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import numpy as np
import pickle

###########global constants#################
FILE_PATH = "purr_scooter_data.csv"
MILESToMETERS = 1609.34
MIN_LAT = 39.630285
MIN_LON = -86.327757
MAX_LAT = 40.08600352951961
MAX_LON = -85.87417197566317
#earh's circumference in meters
EARTH_CIRCUM = 40000000
#gid size in meters 
GRID_SIZE = 250
#Time_increments in seconds (1 hour = 3600 seconds)
TIME_INCREMENT = 3600
TIME_OFFSET = 0
#############################################

def lat_lon_to_x_y(lat,lon):
    dx = (lon-MIN_LON)*EARTH_CIRCUM*math.cos((MIN_LAT+lat)*math.pi/360)/360
    dy = (lat-MIN_LAT)*EARTH_CIRCUM/360
    dx = int(dx/GRID_SIZE)
    dy = int(dy/GRID_SIZE)

    return dx, dy

def date_to_seconds_since_epoch(date):
    dt = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
    
    return dt.timestamp()

def date_to_time_group(date):
    dt = date_to_seconds_since_epoch(date)
    
    return int((dt-TIME_OFFSET)/TIME_INCREMENT)


def remove_invalid_scooter_data(scooter_data):
    #remove scooter data that have a distance of 0
    df_filter = scooter_data[scooter_data[hd.distance] != 0]
    
    #remove null values
    df_filter.dropna()
    
    for col in scooter_data.columns:
        df_filter = df_filter[df_filter[col] != "NaN"]
        
    lat_col = [hd.start_lat,hd.end_lat]
    lon_col = [hd.start_lon, hd.end_lon]
    
    for lat in lat_col:
        df_filter = df_filter[(df_filter[lat] > MIN_LAT) &
                              (df_filter[lat] < MAX_LAT)]
    for lon in lon_col:
        df_filter = df_filter[(df_filter[lon] > MIN_LON) &
                              (df_filter[lon] < MAX_LON)]
    
    return df_filter

#lambdafunc to turn lat lon into x y
pos_lambda_func = lambda x: pd.Series([lat_lon_to_x_y(x[hd.start_lat],x[hd.start_lon])[0],
                                  lat_lon_to_x_y(x[hd.start_lat],x[hd.start_lon])[1],
                                  lat_lon_to_x_y(x[hd.end_lat],x[hd.end_lon])[0],
                                  lat_lon_to_x_y(x[hd.end_lat],x[hd.end_lon])[1]])

time_lambda_func = lambda x: pd.Series([date_to_time_group(x[hd.start_time]),
                                        date_to_time_group(x[hd.end_time])])    
def build_grid_database(df):
    
    lat_lon_columns = [hd.start_lat,hd.start_lon,hd.end_lat,hd.end_lon]
    
    #turn lat lon values into x y coordinates
    df[lat_lon_columns] =  df.apply(pos_lambda_func, axis = 1)
    
    #new column names
    x_y_col  = ['start_x','start_y','end_x','end_y']
    
    #create dict for rename columns
    rename_columns = dict(zip(lat_lon_columns ,x_y_col))
    
    #rename columns
    df.rename(columns = rename_columns, inplace=True)
    
def create_time_groups(df):
    #get start time and end time
    time_cols = [hd.start_time, hd.end_time]
    
    #group start and end times in 1 hour groups
    df[time_cols] =  df.apply(time_lambda_func, axis = 1)
    
    #new column names
    group_time_col = ['start_time_group', 'end_time_group']
    
    #create dict for rename columns
    rename_columns  = dict(zip(time_cols, group_time_col))
    
    #rename columns
    df.rename(columns = rename_columns, inplace=True)
    
def create_trip_db(df):
    odlist = []
    matrix_dim = lat_lon_to_x_y(MAX_LAT,MAX_LON)
    
    df.sort_values('start_time_group', inplace=True)
    
    indexs = df['start_time_group'].index
    j=0
    cuurent_index = indexs[j]
    for i in range(max(df['start_time_group'])):
        coo = coo_matrix(matrix_dim, dtype=np.int8)
        coo = sp.csr_matrix(coo) 
        while (df['start_time_group'][cuurent_index] == i):
            x = df['start_x'][cuurent_index]
            y = df['start_y'][cuurent_index]
            coo[x,y] +=  1
            j += 1
            cuurent_index = indexs[j]
        odlist.append(coo)
            
    
    
def main():  
    
    df_scooter_data = pd.read_csv(FILE_PATH,sep=',')
    
    #update global TIME_OFFSET
    global TIME_OFFSET 
    
    TIME_OFFSET =  date_to_seconds_since_epoch(df_scooter_data[hd.start_time][0])
    
    df_filtered = remove_invalid_scooter_data(df_scooter_data)
    
    columns_needed = [hd.start_time, hd.end_time, hd.start_lat, 
                      hd.start_lon, hd.end_lat, hd.end_lon]
    
    df_filtered = df_filtered[columns_needed]
    
    df_filtered = df_filtered[:100]
    
    build_grid_database(df_filtered)
    
    print("finished grid database")
    
    create_time_groups(df_filtered)
    
    print("finished time groups database")
    
    trip_db = create_trip_db(df_filtered)
    
    df_filtered.to_csv("out.csv", index = False)
    
    with open('test.pkl', 'wb') as sout:
        pickle.dump(trip_db, sout, pickle.HIGHEST_PROTOCOL)
    
    

if __name__ == "__main__":
    main()