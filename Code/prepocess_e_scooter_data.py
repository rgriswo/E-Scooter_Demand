# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:37:53 2023

@author: Ryan's computer
"""
import sys
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
from matplotlib import pyplot as plt
from escooter_config import FILE_PATH, SCOOTER_DATA , GRID_DICT 

###########global constants#################
pickle_name = SCOOTER_DATA 
grid_dict = GRID_DICT 
MILESToMETERS = 1609.34
MIN_LAT = 39.45696255557861
MIN_LON = -86.52754146554861
MAX_LAT = 40.082422163805646
MAX_LON = -85.70699289671025
RADIUS = 0.3
#earh's circumference in meters
EARTH_CIRCUM = 40000000
#gid size in meters 
GRID_SIZE = 250
#Time_increments in seconds (1 hour = 3600 seconds)
TIME_INCREMENT = 3600
TIME_OFFSET = 0
X_Y_DICT = {}
DEMAND_CUTOFF = 100
DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
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
      if len(X_Y_DICT) == 0:
          X_Y_DICT[x_y] = [0,1]
          #    return X_Y_DICT[x_y][0]

      #if x and y has already been assinged a number return number
      if x_y in X_Y_DICT:
          X_Y_DICT[x_y][1] = X_Y_DICT[x_y][1] + 1
        #    return X_Y_DICT[x_y][0]
        #else assigned x and y a new number and add it to dictionary
      else:
            max_unique_id = max([item[0] for item in X_Y_DICT.values()])
            X_Y_DICT[x_y] = [max_unique_id + 1,1]
            #    return X_Y_DICT[x_y][0]

      return dx, dy
    
def set_min_max_lat_lon(lat,lon):

        global MIN_LAT, MIN_LON, MAX_LAT, MAX_LON
        MIN_LON = min(lon)
        MIN_LAT = min(lat)
        
        MAX_LON = max(lon)
        MAX_LAT = max(lat)
        
        
def date_to_seconds_since_epoch(date):
    dt = datetime.strptime(date, DATE_TIME_FORMAT)
    
    return dt.timestamp()

def date_to_time_group_to_date(time_group):
    seconds = ((time_group*TIME_INCREMENT) + TIME_OFFSET)
    
    seconds = datetime.fromtimestamp(seconds)
    
    dt = seconds.strftime(DATE_TIME_FORMAT)
    
    return dt

def date_to_time_group(date):
    #turn date time into seconds tince epoch
    dt = date_to_seconds_since_epoch(date)
    #get group subtracting off offset and dividing by time window size
    return int((dt-TIME_OFFSET)/TIME_INCREMENT)


def z_score(x,u,r):
    return (x-u)/r

def validate_datetime(date_text):
        try:
           datetime.strptime(date_text, DATE_TIME_FORMAT)
           return True
        except:
            return False
        
def validate_datetime_series(date_series):
    return [validate_datetime(i) for i in date_series]

def remove_invalid_scooter_data(scooter_data):
    #get lat lon headers
    lon_col = [hd.start_lon, hd.end_lon]
    lat_col = [hd.start_lat, hd.end_lat]
    lat_lon_col = [lon_col,lat_col]
    
    for col in lat_lon_col:
        scooter_data[col] = scooter_data[col].astype(float)
    
    #remove scooter data that have a distance of 0
    df = scooter_data[scooter_data[hd.distance] != 0]
    
    #remove null values
    df.dropna()
    
    #check if any colums have a null values if so remove it
    for col in scooter_data.columns:
        df = df[df[col].isnull() == False]
    
    #for i in [hd.start_time, hd.end_time]:
    #    df = df[validate_datetime_series(df[i])]
        
    #remove outliers with z scores greater than 3
    for coord_col in lat_lon_col:
        #0 = start 1 = end
        #combine that start and end collums for either lat or lon
        coord_combined = np.concatenate((df[coord_col[0]], df[coord_col[1]]),axis = 0)
        for col in coord_col:
            #calculate the z_score for column using the population mean and std
            z_scores = z_score(df[col],coord_combined.mean(),coord_combined.std())
            df = df[(abs(z_scores) < 3)]
          
    return df

#lambdafunc to turn lat lon into x y
pos_lambda_func = lambda x: pd.Series(np.concatenate([lat_lon_to_x_y(x[hd.start_lat],x[hd.start_lon]),
                                                    lat_lon_to_x_y(x[hd.end_lat],x[hd.end_lon])]))
end_lambda_func = lambda x: pd.Series(lat_lon_to_x_y(x[hd.end_lat],x[hd.end_lon]))

                                      #lat_lon_to_x_y(x[hd.end_lat],x[hd.end_lon])])

time_lambda_func = lambda x: pd.Series([date_to_time_group(x[hd.start_time]),
                                        date_to_time_group(x[hd.end_time])])    
def build_grid_database(df):
    
    replace_columns = [hd.start_lat, hd.start_lon, hd.end_lat, hd.end_lon]
    
    #turn lat lon values into x y coordinates
    x_y_coord =  df.apply(pos_lambda_func, axis = 1)

    df[replace_columns] = x_y_coord

    #new column names
    x_y_col  = ['start_x', 'start_y', 'end_x', 'end_y']
    
    #create dict for rename columns
    rename_columns = dict(zip(replace_columns ,x_y_col))
    
    #rename columns
    df.rename(columns = rename_columns, inplace=True)
    
    global X_Y_DICT
    
    #get high demands cells
    keys = np.array(tuple(X_Y_DICT.keys()))
    
    unique_id = np.array([item[0] for item in X_Y_DICT.values()])
    
    demand_count = np.array([item[1] for item in X_Y_DICT.values()])
    
    #high_demand_mask = (demand_count >= DEMAND_CUTOFF)
    high_demand_mask = (stats.zscore(demand_count) > -3)
    
    high_demand_ids = unique_id[high_demand_mask]
    
    high_demand_keys = keys[high_demand_mask]
    
    row2 = [item[0] for item in X_Y_DICT.values()]
    row3 = [item[1] for item in X_Y_DICT.values()]
    data = {'row_1': X_Y_DICT.keys(), 'row_2': row2, 'row_3': row3}
    
    pd.DataFrame.from_dict(data).to_csv(grid_dict, index = False)
    
    #update dictionary to only include high demand cells
    #X_Y_DICT = dict((tuple(k),X_Y_DICT[tuple(k)]) for k in high_demand_keys)
    
    
    #filter out cells without high demands
    #for col in x_y_col:
    #    df = df[list((i in high_demand_ids for i in df[col]))]
    

    return df              

 
    
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
    
    df['start_time_group'] = df['start_time_group'] - min(df['start_time_group'])
    
def create_trip_db(df):
    total_demand = []
    odlist = []
    
    lats = np.concatenate((df['start_x'], df['end_x']))    
    lons = np.concatenate((df['start_y'], df['end_y']))
    
    lats = np.unique(lats)
    lons = np.unique(lons)
    
    matrix_dim = (len(lats), len(lons))
    
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
        start_coo = coo_matrix(matrix_dim, dtype=np.uint16)
        start_coo = sp.csr_matrix(start_coo) 
        
        end_coo = coo_matrix(matrix_dim, dtype=np.uint16)
        end_coo = sp.csr_matrix(end_coo) 
        
        while (df['start_time_group'][current_pandas_index] == i):
            
            #get x and y
            start_x = np.where(lats == df['start_x'][current_pandas_index])[0]
            start_y = np.where(lons == df['start_y'][current_pandas_index])[0]
            
            end_x = np.where(lats == df['end_x'][current_pandas_index])[0]
            end_y = np.where(lons == df['end_y'][current_pandas_index])[0]
            
            #increment cell demand for x and y coordinates
            start_coo[start_x,start_y] +=  1
            
            end_coo[end_x,end_y] +=  1
            
            #increment to the next index
            j += 1
            
            current_pandas_index = indexs[j]
        timestamp = [date_to_time_group_to_date(i), date_to_time_group_to_date(i+1)]  
        odlist.append([start_coo,end_coo,timestamp])
        total_demand.append(start_coo.sum()+end_coo.sum())
        
    
    
    plt.figure()
    plt.title('Total Demand Graph')
    plt.plot(total_demand)  
    plt.savefig('Total_Demand_Graph.pdf')
    
    return odlist

def format_date_time(df):
    global DATE_TIME_FORMAT 
    if hd.end_date == "none" or hd.start_date == "none":
        DATE_TIME_FORMAT = hd.time_format
        return
    DATE_TIME_FORMAT = hd.date_format + "T" + hd.time_format
    
    df[hd.start_time] = df[hd.start_date] + "T" + df[hd.start_time]
    df[hd.end_time] = df[hd.end_date] + "T" + df[hd.end_time]
    
    return df

def split_date(df,col):
        df[col] = df[col].apply(lambda x: pd.Series(x.split()[0]))
        return df
def main():  
    
    df_scooter_data = pd.read_csv(FILE_PATH,sep=',')
    
    df_scooter_data = split_date(df_scooter_data,hd.start_date)
    df_scooter_data = split_date(df_scooter_data,hd.end_date)
    
    df_scooter_data = format_date_time(df_scooter_data)
    #update global TIME_OFFSET
    global TIME_OFFSET , database 
    
    df_filtered = remove_invalid_scooter_data(df_scooter_data)
    
    #TIME_OFFSET =  date_to_seconds_since_epoch(df_scooter_data[hd.start_time][0])
    
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

    print(MIN_LAT, MAX_LAT)
    print(MIN_LON, MAX_LON)
    
    print("start building grid database")
    
    df_filtered = build_grid_database(df_filtered)
    
    print("finished grid database")
    
    create_time_groups(df_filtered)
    
    print("finished time groups database")
    
    trip_db = create_trip_db(df_filtered)
    
    print("Created trib db")
    
    df_filtered.to_csv("out.csv", index = False)
    
    
    with open(pickle_name, 'wb') as sout:
        pickle.dump(trip_db, sout, pickle.HIGHEST_PROTOCOL)
    
    

if __name__ == "__main__":
    main()