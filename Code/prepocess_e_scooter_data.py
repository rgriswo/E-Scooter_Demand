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
from scipy import stats
import numpy as np
import pickle
import torch

###########global constants#################
FILE_PATH = "C:\\Users\\ryang\\Desktop\\E_Scooter_Demand\\E-Scooter_Demand\\Data\\purr_scooter_data.csv"
MILESToMETERS = 1609.34
MIN_LAT = 39.630285
MIN_LON = -86.327757
MAX_LAT = 40.08600352951961
MAX_LON = -85.87417197566317
RADIUS = 0.3
#earh's circumference in meters
EARTH_CIRCUM = 40000000
#gid size in meters 
GRID_SIZE = 500
#Time_increments in seconds (1 hour = 3600 seconds)
TIME_INCREMENT = 3600
TIME_OFFSET = 0
GRID_DICT = {}
#############################################

def lat_lon_to_x_y(lat,lon):
      #calculate the distance
      dx = (lon-MIN_LON)*EARTH_CIRCUM*math.cos((MIN_LAT+lat)*math.pi/360)/360
      dy = (lat-MIN_LAT)*EARTH_CIRCUM/360
      #divide distance by size
      dx = int(dx/GRID_SIZE)
      dy = int(dy/GRID_SIZE)

      #combine x and y grid location
      x_y = (dx,dy)
      
      #if its the first grid location add it to dictionary
      if len(GRID_DICT) == 0:
          GRID_DICT[x_y] = 0
          return 0
      
      #if x and y has already been assinged a number return number
      if x_y in GRID_DICT:
          return GRID_DICT[x_y]
      #else assigned x and y a new number and add it to dictionary
      else: 
          GRID_DICT[x_y] = max(GRID_DICT.values()) + 1
          return GRID_DICT[x_y]

# def lat_lon_to_x_y(lat,lon):
#         print((lat- MIN_LAT)/RADIUS)
#         x = abs(int((lat - MIN_LAT)/RADIUS))
#         y = abs(int((lon - MIN_LON)/RADIUS))
#         return x, y
    
def set_min_max_lat_lon(lat,lon):

        global MIN_LAT, MIN_LON, MAX_LAT, MAX_LON
        MIN_LON = np.floor(min(lon))
        MIN_LAT = np.floor(min(lat))
        
        MAX_LON = np.ceil(max(lon))
        MAX_LAT = np.ceil(max(lat))
        
        
def date_to_seconds_since_epoch(date):
    dt = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
    
    return dt.timestamp()

def date_to_time_group(date):
    #turn date time into seconds tince epoch
    dt = date_to_seconds_since_epoch(date)
    #get group subtracting off offset and dividing by time window size
    return int((dt-TIME_OFFSET)/TIME_INCREMENT)


def remove_invalid_scooter_data(scooter_data):
    #get lat lon headers
    lat_lon_col = [hd.start_lat, hd.end_lat, hd.start_lon, hd.end_lon]
    
    scooter_data[lat_lon_col] = scooter_data[lat_lon_col].astype(float)
    
    #remove scooter data that have a distance of 0
    df = scooter_data[scooter_data[hd.distance] != 0]
    
    #remove null values
    df.dropna()
    
    #check if any colums have a null values if so remove it
    for col in scooter_data.columns:
        df = df[df[col].isnull() == False]
    
    #remove outliers with z scores greater than 3
    for lat in lat_lon_col:
        df = df[(abs(stats.zscore(df[lat])) < 3)]
        
    
    return df

#lambdafunc to turn lat lon into x y
pos_lambda_func = lambda x: pd.Series([lat_lon_to_x_y(x[hd.start_lat],x[hd.start_lon]),
                                  lat_lon_to_x_y(x[hd.end_lat],x[hd.end_lon])])

time_lambda_func = lambda x: pd.Series([date_to_time_group(x[hd.start_time]),
                                        date_to_time_group(x[hd.end_time])])    
def build_grid_database(df):
    
    lat_lon_columns = [hd.start_lat,hd.end_lat]
    
    #turn lat lon values into x y coordinates
    df[lat_lon_columns] =  df.apply(pos_lambda_func, axis = 1)
    
    #new column names
    x_y_col  = ['start','end']
    
    #create dict for rename columns
    rename_columns = dict(zip(lat_lon_columns ,x_y_col))
    
    #rename columns
    df.rename(columns = rename_columns, inplace=True)
    
    df.drop(columns=[hd.start_lon, hd.end_lon], axis=1, inplace=True)                

    
    
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
    matrix_dim = (matrix_dim, matrix_dim)
    
    #sort data by time groups
    df.sort_values('start_time_group', inplace=True)
    
    #get pandas indexs
    indexs = df['start_time_group'].index
    
    
    #current index
    j=0
    
    #current pandas indedx
    current_pandas_index = indexs[j]

    for i in range(max(df['start_time_group'])):
        
        #create coordinate matrix for time group
        coo = coo_matrix(matrix_dim, dtype=np.int8)
        coo = sp.csr_matrix(coo) 
        
        while (df['start_time_group'][current_pandas_index] == i):
            
            #get x and y
            x = df['start'][current_pandas_index]
            y = df['end'][current_pandas_index]
            

            #increment cell demand for x and y coordinates
            coo[x,y] +=  1
            
            #increment to the next index
            j += 1
            
            current_pandas_index = indexs[j]
            
        odlist.append(coo)
        
    return odlist
            
    
    
def main():  
    
    df_scooter_data = pd.read_csv(FILE_PATH,sep=',')
    
    #update global TIME_OFFSET
    global TIME_OFFSET  
        
    TIME_OFFSET =  date_to_seconds_since_epoch(df_scooter_data[hd.start_time][0])
    
    df_filtered = remove_invalid_scooter_data(df_scooter_data)
    
    #get important columns
    columns_needed = [hd.start_time, hd.end_time, hd.start_lat, 
                      hd.start_lon, hd.end_lat, hd.end_lon]
    
    #remove other comlumns
    df_filtered = df_filtered[columns_needed]
    
    #get list of all lat and longs from start and end groups
    lat = np.concatenate((df_filtered[hd.start_lat], df_filtered[hd.end_lat]))    
    lon = np.concatenate((df_filtered[hd.start_lon], df_filtered[hd.end_lon]))
    
    #set min and max lat lons 
    set_min_max_lat_lon(lat, lon)
    
    build_grid_database(df_filtered)
    
    print("finished grid database")
    
    create_time_groups(df_filtered)
    
    print("finished time groups database")
    
    trip_db = create_trip_db(df_filtered)
    
    df_filtered.to_csv("out.csv", index = False)
    
    data = {'row_1': GRID_DICT.keys(), 'row_2': GRID_DICT.values()}
    
    pd.DataFrame.from_dict(data).to_csv("grid_dict.csv", index = False)
    
    with open('test3.pkl', 'wb') as sout:
        pickle.dump(trip_db, sout, pickle.HIGHEST_PROTOCOL)
    
    

if __name__ == "__main__":
    main()