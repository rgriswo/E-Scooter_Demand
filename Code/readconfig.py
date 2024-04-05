#!/usr/bin/env python 
#
# readconfig.py
# Copyright (C) 2024, Stephen Kim
# 

import configparser
import json

# write a configuration dictionary into a file of type INI
# A configuration dictionary is 2D dictionary.  The first dimension indicates a class, and
# the second dimension is the key of the content. 
# An INI file is a configuration file consisting of a text-based content.  For its detail, 
# refer https://en.wikipedia.org/wiki/INI_file
def write_config(filename, cdict): 
    cobj = configparser.ConfigParser()
        
    for s in cdict.keys():
        cobj.add_section(s)
        
    for s in cdict.keys():
        for k in cdict[s].keys(): 
 #           print("%s - %s - %s" % (s, k, cdict[s][k]))
           if type(cdict[s][k]) is str: 
                cobj.set(s,k, '"'+cdict[s][k].replace("%", "%%")+'"')
           else:
                cobj.set(s,k,str(cdict[s][k]))
                
    with open(filename, "w") as fout:
        cobj.write(fout)

# read a configuration INI file and return it as a dictionary, such as
# config['class']['key'] = value
#
# If the value is a string, it must be enclosed by a pair of double quotes.
# A percent sign in the string must be escaped by using an addtional percent sign.
def read_config(filename):
    cobj = configparser.ConfigParser()
    with open(filename, "r") as fin:
        cobj.read_file(fin)
        
    cdict = dict()
    for s in cobj.sections():
        items = cobj.items(s)
        cdict[s] = dict(items)
        for k in cdict[s].keys():
#            print("%s - %s - %s" % (s, k, cdict[s][k]))
            cdict[s][k] = json.loads(cdict[s][k])

    return cdict

if __name__ == "__main__": 
    config_filename = "lx.ini"

    # write_config(config_filename)
    config_dict = read_config("lx.ini")
    write_config("lx2.ini", config_dict)
    
    with open("lx2.ini", "r") as fin:
        print(fin.read())
        
