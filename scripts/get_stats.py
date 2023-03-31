from src_util import util_reading
import pathlib
import pandas as pd
from PIL import Image
import numpy as np
from collections import Counter
import os

def get_shape(config_json):
    """
    It generates two csv files: csv files
    * Columns: filename and shape
    * Columns: shape and quantity
    """

    df_shape = pd.DataFrame(columns = ["filename","filepath","shape"])
    type_file  = config_json["type_file"]
    pathlib_directory = pathlib.Path(config_json["directory_data"])

    pathlib_output = pathlib.Path(config_json["directory_output"])
    os.makedirs(pathlib_output, exist_ok = True)

    for path_image in pathlib_directory.rglob(f"*.{type_file}"):
        img = Image.open(path_image)
        a_img = np.array(img)
        # print(a_img.shape)
       
        df_temp = pd.DataFrame({"filename":[path_image.name],
                                "filepath":[str(path_image)],
                                "shape":[a_img.shape]})
            
        df_shape = pd.concat([df_shape, df_temp], ignore_index=True)


    filename_shape = f"{config_json['suffix']}_shape_list.csv"
    path_file = pathlib_output.joinpath(filename_shape)
    df_shape.to_csv(path_file)

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

    print("Hello World")


def main():

    args = util_reading.get_parser()
    config_json = util_reading.get_json(args)


    get_shape(config_json)
    # filepath = config_json["list_filenames"][0]

    # df_message_one = pd.read_csv(filepath, index_col = 0)  

    # list_criteria = ["app","apps","application"]
    # message = "The app is the solution to your problems"
    # get_boolean_criteria(message, list_criteria, config_json)
    # message = "Stand up"
    # list_criteria = ["less-than-six-words"]
    # get_boolean_criteria(message, list_criteria)
    # print(config_json)
    # apply_criteria(config_json)

if __name__ == "__main__":
    main()