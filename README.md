# Property-directed Generative Design of Inorganic Materials 
Multi-property-directed WyCryst framework with Wyckoff data augmentation and transfer learnig 


**Maintainer**: [Shuya Yamazaki](https://github.com/shuyayamazaki) [email: shuya001@e.ntu.edu.sg] 

**Original WyCryst repo**: [WyCryst](https://github.com/RaymondZhurm/WyCryst) @[RaymondZhurm](https://github.com/RaymondZhurm)

## Installation
Please use python 3.8.10 to run the model.

Follow the steps below to install all required packages:
 ```
conda install -r requirement.txt
 ```

## Data 
Please find `df_allternary_newdata.pkl` from the [WyCryst](https://github.com/RaymondZhurm/WyCryst) `Data release` and put it in the `data` directory

## Usage 
Please provide `PATH_TO_DATA` and `PATH_TO_TEMP_FILES` in `tain.py` and `generate.py`

You need to run `tain.py` before you perform `generate.py`

## Training the MPVAE model
To train the MPVAE model, run:
 ```
python train.py
 ```

## Property-directed generation 
For property-directed generation (e.g. Ef < -0.5 & Eg ~= 1.5), run:
 ```
python generate.py
 ```
