import pandas as pd
import numpy as np
import os


"""
Read in the file
For each line, break the line into key value pairs.
Determine what information needs to be "kept" and stored.
If we want to keep it, save to a dictionary.
Convert that dictionary to a dataframe and save as a csv.
"""


def process_lines(line, data_dict):
    # split the line and add it to the dictionary we are creating
    if 'year' not in data_dict:
        data_dict['year'] = []
        data_dict['pop'] = []
    line_arr = line.split('=')
    if line_arr[0].isdigit():
        data_dict['year'].append(line_arr[0])
        data_dict['pop'].append(line_arr[1])
    else:
        key, val = line_arr[0], line_arr[1]
        data_dict[key] = val
    return data_dict


def filter_dict(data_dict):
    # filters out unwanted key value pairs
    wanted_keys = {'id', 'Species', 'Decimal Latitude', 'Decimal Longitude',
                   'Terrestrial biome', 'Terrestrial realm', 'Sampling method',
                   'year', 'pop', 'Units'}
    unwanted_keys = set(data_dict) - wanted_keys
    for unwanted_key in unwanted_keys:
        del data_dict[unwanted_key]
    return data_dict


def fill_in_rows(data_dict):
    #duplicate non-year and non-population values for other keys
    keys = set(data_dict)
    desired_length = len(data_dict['year'])
    for key in keys:
        if key not in {'year', 'pop'}:
            data_dict[key] = [data_dict[key]] * desired_length
    return data_dict


def convert_to_dataframe(data_dict, df):
    #building the pandas dataframe of population data from the open source data
    if df is None:
        df = pd.DataFrame(data_dict)
    else:
        temp_df = pd.DataFrame(data_dict)
        df = df.append(temp_df, ignore_index=True, sort=True)
    return df


def process_file(input_file, df):
    #read in one file and output a data frame of that file
    f = open(input_file, "r")
    lines = f.read().splitlines()
    data_dict = {}
    for line in lines:
        data_dict = process_lines(line, data_dict)
    data_dict = filter_dict(data_dict)
    data_dict = fill_in_rows(data_dict)
    output_df = convert_to_dataframe(data_dict, df)
    f.close()
    return output_df


def process_all_files(final_path):
    path = '../../Raw/' + final_path + '/'
    df = None
    for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isfile(current):
            df = process_file(current, df)
    output_file_name = '../../processed/' + final_path + '.csv'
    df.to_csv(output_file_name, index=False)




if __name__ == '__main__':
    # finalPath = "Bison_US"
    finalPathList = ["Coyote_US", "DesertTortoise_US", "GreyWolf_US"
                    , "Raccoon_US", "RedFox_US", "WhiteTailedDeer_US"]
    for finalPath in finalPathList:
        process_all_files(finalPath)
