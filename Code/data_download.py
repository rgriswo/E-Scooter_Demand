# -*- coding: utf-8 -*-
"""
Created on Thu May 25 00:12:19 2023

@author: dskim
"""

import os, sys, hashlib
from tqdm import tqdm
import requests
from typing import Any, Optional

def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)

def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)
    
def download_file(url, target, md5=None): 
    if not os.path.isfile(target): 
        print("downloading the file...")
        response = requests.get(url, stream=True)
        with open(target, "wb") as handle:
            for data in tqdm(response.iter_content(chunk_size=1024*1024)):
                handle.write(data)
            
    if md5 == None or check_integrity(target, md5):         
        n = os.path.getsize(target)
        print("The file '%s' is available (%d bytes)" % (target, n))
        return True
    
    print("\nThe file has some problem\n", target)
    return False
            
data_db = {'kansas-scooter': 
              {'addr':"http://et.engr.iupui.edu/~dskim/downloadable/data/", 
               'name':"Kansas-Microtransit__Scooter_and_Ebike__Trips.csv", 
               'md5': '87ad7e73db0f4015cf04d9f965a88943'
              },
           'indianapolis-scooter': 
              {'addr':"http://et.engr.iupui.edu/~dskim/downloadable/data/", 
               'name':"purr_scooter_data.csv", 
               'md5': 'de155a9f3cd9cc434cf49a6d104d3c0e'
              }
          }
    
def ev_catalog():
    return list(data_db.keys())

def ev_download(key, target_path):
    if key in ev_catalog():
        source = os.path.join(data_db[key]['addr'], data_db[key]['name'])
        destin = os.path.join(target_path, data_db[key]['name'])
        # print(source)
        # print(destin)
        # print(data_db[key]['md5'])
        rval = download_file(source, destin, data_db[key]['md5'])
        return rval
    else:
        return False
        
if __name__ == '__main__':    
    destin_path = r"C:\Users\dskim\Documents\Data\scooter"
    key = "kansas-scooter"
#    key = "indianapolis-scooter"    
    ev_download(key, destin_path)
      