"""
Data augmentation for WyCryst+ Framework
"""

import re
import json
from tqdm import tqdm
import pandas as pd
from os.path import join


def wyckoff_augment(df):
    
    def extract_letters(input_dict):
        letters = re.compile(r'[a-zA-Z]+')
        extracted_letters = [re.findall(letters, value) for values in input_dict.values() for value in values]
        return [item for sublist in extracted_letters for item in sublist]

    def create_compatibility_dict(initial_list, transformed_list):
        return dict(zip(initial_list, transformed_list))

    def generate_all_possible_transformations(input_list, compatibility_dicts):
        return [list(map(compatibility_dict.get, input_list, input_list)) for compatibility_dict in compatibility_dicts]

    def process_input_string(input_string):
        input_lists = input_string.split(', ')
        initial_list = input_lists[0].strip('[]').split()
        compatibility_lists = [lst.strip('[]').split() for lst in input_lists[1:]]
        compatibility_dicts = [create_compatibility_dict(initial_list, transformed_list) for transformed_list in compatibility_lists]
        return initial_list, compatibility_dicts

    def process_input(input_dict, list1, list2):
        output_dict = {element: [value[:-1] + list2[list1.index(value[-1])] for value in values if value[-1] in list1] for element, values in input_dict.items()}
        return output_dict

    # Load Wyckoff sets
    with open(join(module_dir,'wyckoff_sets.json')) as f:
        wyckoff_sets = json.load(f)
    wyckoff_sets = {int(key): value for key, value in wyckoff_sets.items()}

    df = df.reset_index()
    rows_to_add = []

    for n in tqdm(range(len(df))):
        x = df.at[n, 'wyckoff_dic']
        wyckoff_let = extract_letters(x)
        sgn = df.at[n, 'spacegroup_number']
        input_string = wyckoff_sets.get(sgn)
        
        if input_string:
            initial_list, compatibility_dicts = process_input_string(input_string)
            all_possible_outputs = generate_all_possible_transformations(wyckoff_let, compatibility_dicts)

            for output in all_possible_outputs:
                output_dict = process_input(x, wyckoff_let, output)
                row_to_duplicate = df.iloc[n].copy()
                row_to_duplicate['wyckoff_dic'] = output_dict
                rows_to_add.append(row_to_duplicate)

    if rows_to_add:
        new_rows_df = pd.DataFrame(rows_to_add)
        df = pd.concat([df, new_rows_df], ignore_index=True)
        
    # Drop duplicates
    df['wyckoff_str'] = df['wyckoff_dic'].apply(str)
    df = df.drop_duplicates(subset=['wyckoff_str', 'ind'])
    df = df.drop(columns=['wyckoff_str'])

    return df