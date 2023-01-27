from ast import arg
from unittest import result
import pandas as pd
import argparse
import os
from diskcache import Index



if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="", help="experiments cache directory", required=True)
    # parser.add_argument("--output_csv", type=str, default="", help="output csv file", required=True)
    args =  parser.parse_args()  
    assert args.cache_dir.endswith("/"), "cache_dir should end with '/'" 
    assert os.path.exists(
        args.cache_dir), "cache dir does not exist"

    
    print(">> Loading cache")
    
    cache = Index(args.cache_dir)

    resent_keys  = []
    densenet_keys = []


    
    
    for key in cache.keys():
        print(key)
        if "resnet" in key:
            resent_keys.append(key)
        elif "densenet" in key: #  and  "femnist" not in key:
            densenet_keys.append(key)
        # else:
        #     raise ValueError(f"{key}")
    

    resent_keys.reverse()
    densenet_keys.reverse()

    with open("keys_resnet.txt","w") as f:
        for k in resent_keys:
            f.write(k + "\n")

    with open("keys_densenet.txt","w") as f:
        for k in densenet_keys:
            f.write(k + "\n")

