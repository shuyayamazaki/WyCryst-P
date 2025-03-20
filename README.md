# Multi-property directed Generative Design of Inorganic Materials 
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
Please find `df_allternary_newdata.pkl` from the [WyCryst](https://github.com/RaymondZhurm/WyCryst) `Data release` (or use your own preprocessed data) and place it into the `data` directory

## Usage 
Before running the scripts, please specify the following paths in both `train.py` and `generate.py`:

`PATH_TO_DATA`: Path to your dataset directory

`PATH_TO_TEMP_FILES`: Directory where trained weights and temporary files will be stored

## Training the MPVAE model
To train the MPVAE model on your own dataset, run:
 ```
python train.py
 ```
Expected run time: 10-20 minutes

## Property-directed generation 
To perform property-directed generation using the pre-trained MPVAE, run:
 ```
python generate.py
 ```
Expected Output:
The script `generate.py` outputs a CSV file containing sampled Wyckoff genes after applying the filtering criteria. The resulting CSV will be saved in the directory specified by `PATH_TO_TEMP_FILES`. Expected run time is around 20-30 minutes (depending on the generation size).
