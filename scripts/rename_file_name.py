from src_util import util_reading
import pathlib
import pandas as pd
from PIL import Image
import numpy as np
from collections import Counter
import os
import re

def rename_files(config_json):
    """
    It allows to change the filename from one string to another
    """

    l_old_name =  config_json["list_old_name"]
    l_new_name = config_json["list_new_name"]
    
    pathlib_directory = pathlib.Path(config_json["directory_data"])
    type_file  = config_json["type_file"]

    for path_image in pathlib_directory.rglob(f"*.{type_file}"):
        old_name = path_image.name
        
        # determine presence of old name in filename
        for i, old_name_value in enumerate(l_old_name):
            if old_name_value in old_name:
                parent_path = path_image.parent
                new_name = old_name.replace(old_name_value, l_new_name[i])

                new_path = parent_path.joinpath(new_name)
                path_image.rename(new_path)

def main():

    args = util_reading.get_parser()
    config_json = util_reading.get_json(args)

    rename_files(config_json)
    
if __name__ == "__main__":
    main()