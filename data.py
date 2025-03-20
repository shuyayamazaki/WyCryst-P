"""
Data loader for WyCryst+ Framework
"""
from featurizer import *
import numpy as np
import re
from os.path import abspath, dirname, join
import json

module_dir = './data/' #PATH_TO_DATA
module_dir2 = './temp_files/' # PATH_TO_TEMP_FILES

def wyckoff_para_loader():

    with open(join(module_dir, "wyckoff-position-multiplicities.json")) as file:
        wyckoff_multiplicity_dict = json.load(file)
    with open(join(module_dir, "wyckoff-position-params.json")) as file:
        param_dict = json.load(file)

    wyckoff_multiplicity_array = np.zeros((231, 27))
    for sg in wyckoff_multiplicity_dict:
        for wp in wyckoff_multiplicity_dict[sg]:
            if wp.isupper():
                site_num = ord(re.sub('[^a-zA-Z]+', '', wp)) - 39
            if wp.islower():
                site_num = ord(re.sub('[^a-zA-Z]+', '', wp)) - 97
            wyckoff_multiplicity_array[int(sg), site_num] = wyckoff_multiplicity_dict[sg][wp]
    wyckoff_multiplicity_array = wyckoff_multiplicity_array[:, :-1]

    wyckoff_DoF_array = np.zeros((231, 27))
    for sg in param_dict:
        for wp in param_dict[sg]:
            if wp.isupper():
                DoF_num = ord(re.sub('[^a-zA-Z]+', '', wp)) - 39
            if wp.islower():
                DoF_num = ord(re.sub('[^a-zA-Z]+', '', wp)) - 97
            wyckoff_DoF_array[int(sg), DoF_num] = param_dict[sg][wp]
    wyckoff_DoF_array = wyckoff_DoF_array[:, :-1]

    return wyckoff_multiplicity_array, wyckoff_DoF_array

    
def get_input_df():
    df_all = pd.read_pickle(join(module_dir,'df_allternary_newdata.pkl'))
    df = df_all[df_all['nsites'] <= 20]
    df_clean = df[['formation_energy_per_atom', 'band_gap', 'e_above_hull', 'pretty_formula', 'spacegroup.crystal_system',
                   'spacegroup.number', 'wyckoff_dic', 'is_stable', 'cif']]
    df_clean.rename(columns={'spacegroup.crystal_system': 'spacegroup_crystal_system'
        , 'spacegroup.number': 'spacegroup_number'}, inplace=True)
    df_clean['ind'] = [i for i in range(len(df_clean))]
    df_clean = df_clean[df_clean['e_above_hull'] < 0.1]
    df_clean = df_clean[df_clean['formation_energy_per_atom'] <= 1]

    df_clean['icsd_check'] = df_clean['is_stable'] * 1

    return df_clean

