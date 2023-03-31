# About The Project

This project will allow to get some statistics before preprocessing your images.

# Getting Started

## Installation

In order to be able to use src_util functions. You need to install from parent directory.
```bash
pip install -e .
```

## Usage

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


## Contact

Paul Calle - pcallec@ou.edu
