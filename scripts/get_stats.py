"""
get_stats.py

Based on the configuration file.
It looks for files with type `type_file` in the folder `directory_data`.
The script outputs two dataframes in `directory_output`.
* Dataframe with columns: filename, filepath, shape
* Dataframe with columns shape and its respective quantity
"""

from src_util import util_reading
import pathlib
import pandas as pd
from PIL import Image
import numpy as np
from collections import Counter
import os
import re

def get_shape(config_json):
    """
    It generates two csv files: csv files
    * Columns: filename and shape
    * Columns: shape and quantity
    """
    # Dataframe with columns: filename, filepath, shape
    df_shape = pd.DataFrame(columns = ["filename","filepath","shape"])
    
    # Search file with specific file type e.g png, jpeg
    type_file  = config_json["type_file"]
    pathlib_directory = pathlib.Path(config_json["directory_data"])
    pathlib_output = pathlib.Path(config_json["directory_output"])
    os.makedirs(pathlib_output, exist_ok = True)

    # Loop through each image 
    for path_image in pathlib_directory.rglob(f"*.{type_file}"):
        img = Image.open(path_image)
        a_img = np.array(img)
        # print(a_img.shape)
       
        df_temp = pd.DataFrame({"filename":[path_image.name],
                                "filepath":[str(path_image)],
                                "shape":[a_img.shape]})
            
        df_shape = pd.concat([df_shape, df_temp], ignore_index=True)


    # filename for the dataframe as csv
    filename_shape = f"{config_json['suffix']}_shape_list.csv"
    path_file = pathlib_output.joinpath(filename_shape)
    df_shape.to_csv(path_file)

    # Create dataframe with summary of shapes
    # dataframe with columns shape and its respective quantity
    df_shape_summary = pd.DataFrame(columns = ["shape", "qty"])

    counter_shape = Counter(df_shape["shape"])
    d_counter_shape = dict(counter_shape)
        
    for key, val in d_counter_shape.items():
        df_temp = pd.DataFrame({"shape":[key], 
                                "qty":[val]})

        df_shape_summary = pd.concat([df_shape_summary, df_temp], ignore_index=True)

    filename_shape_summary = f"{config_json['suffix']}_shape_summary_list.csv"
    path_file = pathlib_output.joinpath(filename_shape_summary)
    df_shape_summary.to_csv(path_file)

def main():
    args = util_reading.get_parser()
    config_json = util_reading.get_json(args)

    get_shape(config_json)

if __name__ == "__main__":
    main()