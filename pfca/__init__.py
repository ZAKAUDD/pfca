#File System Architecture

import os
import sys
import datetime
import pandas as pd


global cur_path, raw_data_dir, nifti_dir, params_dir
#filesystem for the package
cur_path = os.getcwd()
print("Current Working Directory: "+ cur_path)
try:
    os.makedirs(cur_path+'/params')
    os.makedirs(cur_path+'/visuals/3d_vtk')
    os.makedirs(cur_path+'/visuals/stills_2d')
    os.makedirs(cur_path+'/output')
    os.makedirs(cur_path+'/datasets')
    os.makedirs(cur_path+'/datasets/raw')
    os.makedirs(cur_path+'/datasets/nifti')
    print("Directory Structure created successfully.")
except:
    print("The file structure creation failed. Structure already exists!")

def init_path():
    #input variables...to be saved inside the input_params.csv
    if os.path.isfile(cur_path+'/params/input_params.csv'):
        file = pd.read_csv(cur_path+'/params/input_params.csv')
        raw_data_dir = file['raw'].values[0]
        nifti_dir = file['nifti'].values[0]
        params_dir = cur_path+'/params/input_params.csv'
        return raw_data_dir, nifti_dir
    else:
        raw_data_dir = str(cur_path)+"/datasets/raw/"
        nifti_dir = str(cur_path)+"/datasets/nifti/"
        d_modified = datetime.date.today()
        file_dataframe = {'raw': [raw_data_dir], 'date_added': [d_modified], 'nifti': [nifti_dir], 'cur_path' : [cur_path]}
        r = pd.DataFrame(file_dataframe)
        r.to_csv(cur_path+'/params/input_params.csv')
        return raw_data_dir, nifti_dir
