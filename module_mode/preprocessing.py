import numpy as np
import skimage
import scipy.io
import pydicom
import os
import sys
import nibabel as nb
import pandas as pd
import datetime
from pylab import *
import ants
from time import time


#calculate affine matrix for dicom to nifti conversion
#Please refer the following useful paper to understand the process:
'''
Li, Xiangrui, et al. "The first step for neuroimaging data analysis: DICOM to NIfTI conversion." 
Journal of neuroscience methods 264 (2016): 47-56.
'''

def dcm_affine(file_list):
    #Arguments -->
    #file_list : List of directories of all files in a volume
    #Note: Make sure the file list is correct and have all the slices of the volume
    
    #First, we will find the first and last slice in the volume
    for leaf in file_list:
        temp = pydicom.read_file(leaf)
        if int(temp.InstanceNumber) == len(file_list):
            last_slice = leaf
        if int(temp.InstanceNumber) == 1 :            
            first_slice = leaf            
    #Get the direction cosines for the row and column transformation
    [rx, ry, rz, cx, cy, cz] = [float(i) for i in ((pydicom.read_file(first_slice)).ImageOrientationPatient)]
    #Get the ImagePositionPatient of the first slice of volume
    [x1, y1, z1] = [float(i) for i in ((pydicom.read_file(first_slice)).ImagePositionPatient)]
    #Get the ImagePositionPatient of the last slice of volume    
    [xn, yn, zn] = [float(i) for i in ((pydicom.read_file(last_slice)).ImagePositionPatient)]
    #Get spacing of each individual voxel
    [vr, vc] = [float(i) for i in ((pydicom.read_file(last_slice)).PixelSpacing)]
    
    #Affine matrix for the conversion
    affine_matrix = np.array([[rx*vr, cx*vc, ((xn-x1)/(len(file_list)-1)), x1],
                             [ry*vr, cy*vc, ((yn-y1)/(len(file_list)-1)), y1],
                             [rz*vr, cz*vc, ((zn-z1)/(len(file_list)-1)), z1],
                             [0, 0, 0, 1]])
    
    return affine_matrix


#dicom to nifti conversion
def dcm_to_nifti(image_array, patient_name, nifti_path, file_list = None, rotation = None):
    #Arguments -->
    #file_list    : list of path of all dicom files in the slice
    #image_array  : 3D numpy array of image volume
    #nifti_path   : directory path for storing nifti files
    #patient_name : Alias given to the patient
    import nibabel as nb
    import scipy
    if type(file_list).__name__ == 'NoneType':
        img = nb.Nifti1Image(image_array, affine = np.eye(4))
        nb.save(img, nifti_path + patient_name + "_arbitrary")
        print(patient_name + " nifti file generated successfully.")
        print("Warning: Metadata was not provided. So, the conversion might not be accurate.")
    elif type(file_list).__name__ == 'list':
        #calculate the affine matrix for the dicom to nifti conversion
        affine_mat = dcm_affine(file_list)
        if type(rotation).__name__ != 'NoneType':
            image_arr = np.rot90(image_array, k = 1, axes = (1,0))
        im = nb.Nifti1Image(image_arr, affine = affine_mat)
        nb.save(im, nifti_path + patient_name + "_conv")
        print(patient_name + " nifti file generated successfully with the metadata provided.")        


#FSL Brain Extraction Tool(BET) for skull stripping
#parameters:
#path_nifti = Path where nifti files will be stored
#patient = name of the patient file
def skull_stripping(path_nifti,patient):
    from nipype.interfaces import fsl 
    import gzip
    import shutil    
    nii_file = patient + '_unskulled_anat.nii'
    btr = fsl.BET()
    btr.inputs.in_file = path_nifti + patient + '.nii'
    btr.inputs.frac = 0.7
    btr.inputs.out_file = path_nifti + nii_file
    btr.cmdline
    res = btr.run()
    print('Skull stripping done successfully.')
    with gzip.open(path_nifti + nii_file + '.gz', 'rb') as f_in:
        with open(path_nifti + nii_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    try:
        os.remove(path_nifti + nii_file + '.gz')
        print("Skull stripped file saved successfully!")
    except:
        print('Some error occured while saving the file.')



#Non rigid registration of the patient brain to mni space
#mni template citiation:
#VS Fonov, AC Evans, K Botteron, CR Almli, RC McKinstry, DL Collins and BDCG, 
#Unbiased average age-appropriate atlases for pediatric studies, NeuroImage,Volume 54, Issue 1, January 2011, 
#ISSN 1053–8119, DOI: 10.1016/j.neuroimage.2010.07.033
#
#VS Fonov, AC Evans, RC McKinstry, CR Almli and DL Collins, Unbiased nonlinear average age-appropriate brain 
#templates from birth to adulthood, NeuroImage, Volume 47, Supplement 1, July 2009, 
#Page S102 Organization for Human Brain Mapping 2009 Annual Meeting, 
#DOI: http://dx.doi.org/10.1016/S1053-8119(09)70884-5
#
def mni_template_registration(cur_path, patient_image, patient_name):
    import nibabel as nb
    import ants
    if not os.path.isfile(cur_path + '/datasets/nifti/'+ patient_name + '_mni_registered.nii'):        
        fixed = ants.image_read(cur_path + '/datasets/mni_t2_template.nii')
        moving = patient_image
        fixed.plot(overlay = moving, title = 'Before Registration')
        mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform='SyN')
        warped_moving = mytx['warpedmovout']
        fixed.plot(overlay=warped_moving,
                   title='After Registration')
        nib_image = warped_moving.to_nibabel()
        nb.save(nib_image, cur_path + '/datasets/nifti/'+ patient_name + '_mni_registered.nii')
        return warped_moving
    else:
        warped = ants.image_read(cur_path + '/datasets/nifti/'+ patient_name + '_mni_registered.nii')
        return warped        

















