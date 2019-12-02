import numpy as np
import skimage
import scipy.io
import pydicom
import os
import sys
import datetime
from pylab import *

from time import time

def dcm_lst(subject_folder):
    #Function to develop list of DICOM files present in a directory.
    #Filename should start with a single alphabet followed by a digit i.e. 'Z01', 'Z11' 
    lstDCM = []
    if not os.path.exists(subject_folder):
        print("Subject Path not found. Exiting..")
        sys.exit()
    for dirname, subdirList, fileList in os.walk(subject_folder):
        for filename in fileList:
            if filename[1].isdigit():
                lstDCM.append(os.path.join(dirname,filename))
    return lstDCM  


#Extract 3D image array based on the subject data folder
def dcm_array(subject_folder, orientation = None):
    lst_DCM = dcm_lst(subject_folder)
    #print(lst_DCM)
    refSlice = pydicom.read_file(lst_DCM[0])
    print("Subject Name(For research purposes) :" + str(refSlice.PatientName))
    pixel_dim = (int(refSlice.Rows),int(refSlice.Columns),int(len(lst_DCM)))
    print("Array Dimensions:")
    print(pixel_dim)
    array_dicom = np.zeros(pixel_dim,dtype = refSlice.pixel_array.dtype)
    for filename in lst_DCM:
        da = pydicom.read_file(filename)
        array_dicom[:,:,(int(da.InstanceNumber)-1)] = da.pixel_array
    return array_dicom    


#Extract 3D image array simply based on file list 
def extract_dcm_array(file_list, orientation = None):
    #print(lst_DCM)
    lst_DCM = file_list
    refSlice = pydicom.read_file(lst_DCM[0])
    print("Subject Name(For research purposes) :" + str(refSlice.PatientName))
    pixel_dim = (int(refSlice.Rows),int(refSlice.Columns),int(len(lst_DCM)))
    print("Array Dimensions:")
    print(pixel_dim)
    array_dicom = np.zeros(pixel_dim,dtype = refSlice.pixel_array.dtype)
    for filename in lst_DCM:
        da = pydicom.read_file(filename)
        array_dicom[:,:,(int(da.InstanceNumber)-1)] = da.pixel_array
    return array_dicom 


#to get the list of individual patient folder inside the raw data folder
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))] 
            

#performing operations on phase image
#function to first get the phase and eSWAN image from the heap of different kind of images
def get_label_based_list(file_list, label):
    #The function looks up for Sequence Description in MetaData 
    #Look into metadata label :  (0008, 103e) Series Description   
    #Arguments --->
    #file_list : directory path list of all dicom files obtained using dcm_lst() function
    #label : Sequence Description string which is to be looked inside the sequences   
    label_based_list = []
    import pydicom
    for file in file_list:
        #print("Reading file: " + file)
        temp = pydicom.read_file(file)
        if label == temp.SeriesDescription:
            label_based_list.append(file)
    return label_based_list        

    
