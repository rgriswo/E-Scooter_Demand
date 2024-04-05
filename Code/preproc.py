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
from matplotlib import pyplot as plt
MIN_LAT = 39.45696255557861
MIN_LON = -86.52754146554861
MAX_LAT = 40.082422163805646
MAX_LON = -85.70699289671025

###########global constants#################
pickle_name = "e_scooter_indy.pkl"
grid_dict = 'grid_dict_250_indy.csv'
#gid size in meters 
#Time_increments in seconds (1 hour = 3600 seconds)
TIME_INCREMENT = 3600
TIME_OFFSET = 0
X_Y_DICT = {}
DEMAND_CUTOFF = 100
EARTH_RADIUS = 6371         # in kilometer
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
    return ((min_lat,max_lat),(min_lon,max_lon))

def lat_lon_to_x_y(lat,lon):
      #calculate the distance
      dx = (lon-MIN_LON)*EARTH_CIRCUM*math.cos((MIN_LAT+lat)*math.pi/360)/360
      dy = (lat-MIN_LAT)*EARTH_CIRCUM/360
      #divide distance by size
      dx = int(dx / config['general']['cell_size'])
      dy = int(dy / config['general']['cell_size'])
      
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

#lambdafunc to turn lat lon into x y
pos_lambda_func = lambda x: pd.Series(np.concatenate([lat_lon_to_x_y(x[hd.start_lat],x[hd.start_lon]),
                                                    lat_lon_to_x_y(x[hd.end_lat],x[hd.end_lon])]))
end_lambda_func = lambda x: pd.Series(lat_lon_to_x_y(x[hd.end_lat],x[hd.end_lon]))

                                      #lat_lon_to_x_y(x[hd.end_lat],x[hd.end_lon])])

time_lambda_func = lambda x: pd.Series([date_to_time_group(x[hd.start_time]),
                                        date_to_time_group(x[hd.end_time])])  
    
def build_grid_database(df, bb):    
    replace_columns = ['start_lat', 'start_lon', 'end_lat', 'end_lon']
    
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

# Open database have have different schema.  
#  
def formalize(df, kind):
    if kind == "TYPE_PURR":
        df['start_tag'] = df['start_time_utc'].apply(lambda x: pd.Timestamp(x))
        df['end_tag']   = df['end_time_utc'].apply(lambda x: pd.Timestamp(x))
    elif kind == "TYPE_LOUISVILLE":
        df['start_tag'] = df[['StartDate', 'StartTime']].agg('T'.join, axis=1).apply(lambda x: pd.Timestamp(x))
        df['end_tag'] = df[['EndDate', 'EndTime']].agg('T'.join, axis=1).apply(lambda x: pd.Timestamp(x))
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
    return df


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

from readconfig import read_config

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
    
    df = calculate_time_index(df)
    print("Shape after calculate time index \t", df.shape)

    #get list of all lat and longs from start and end groups
    print("start building grid database")
    bbox = find_bbox(df) 
    
    df = build_grid_database(df, bbox)
    
    return df   
  
    print("finished grid database")
      
    trip_db = create_trip_db(df)
    
    print("Created trib db")
    
    df.to_csv("out.csv", index = False)
    
    
    with open(pickle_name, 'wb') as sout:
        pickle.dump(trip_db, sout, pickle.HIGHEST_PROTOCOL)
    
    

if __name__ == "__main__":
    config = read_config('escooter.ini')
    df = main('purr.csv')
    # df = main('louisville.csv')
    # df = main('kansas.csv')
    # df = main(r'C:\Users\dskim\Documents\Data\scooter\purr_scooter_data.csv')
