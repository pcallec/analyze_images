# About The Project

This project will allow to get some statistics and useful preprocessing utilities before using your images in ML models.

# Getting Started

## Installation

In order to be able to use src_util functions. You need to install from parent directory.
```bash
pip install -e .
```

## Usage

### `get_stats.py`
e.g.
```bash
python get_stats.py -j get_stats_AIIMS.json
```

e.g. 
get_stats_AIIMS.json
```json
{
    "directory_data": "/home/pcallec/analyze_images/data/LighOCT/Dataset/AIIMS_Dataset",
    "type_file":"bmp",
    "directory_output": "/home/pcallec/analyze_images/results",
    "suffix": "data_AIMMS"
}
```
* `directory_data`: path of data
* `type_file`: bmp. jpeg, png
* `directory_output`: path of output directory
*  `suffix`: Suffix to name the output csv files


example shape list
```csv
,filename,filepath,shape
0,Default_0008_Mode3D_0034.bmp,/home/pcallec/analyze_images/data/LighOCT/Dataset/AIIMS_Dataset/Cancer Sample/Patient c11/Default_0008_Mode3D_0034.bmp,"(441, 245, 3)"
1,Default_0007_Mode3D_0089.bmp,/home/pcallec/analyze_images/data/LighOCT/Dataset/AIIMS_Dataset/Cancer Sample/Patient c11/Default_0007_Mode3D_0089.bmp,"(442, 245, 3)"
...
```
example shape list summary
```csv
,shape,qty
0,"(441, 245, 3)",5775
1,"(442, 245, 3)",3465
...
```

### `rename_file_name.py`

```json
{
    "directory_data": "/home/pcallec/mif_outer/data/pig_kidney_subset",
    "type_file": "jpg",
    "list_old_name":["pelvis_calyx"],
    "list_new_name":["pelvis-calyx"]
}
```

* `directory_data`: path of data
* `type_file`: type of file to change name
* `list_old_name`: list of string that if found in file would be modified
*  `list_new_name`: list of string that would be used to rename.

e.g.
```bash
python rename_file_name.py -j rename_pig_kidney_subset.json
```

## Contact

Paul Calle - pcallec@ou.edu
