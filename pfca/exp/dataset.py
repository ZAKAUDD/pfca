#Contains functions to manipulate, augment and manage the data for training the various classifiers available.
#This ranges from functions for generating non-target image patches to handling all the data efficiently in a Pandas Dataframe


#snip a lot of non-microbleeds images from the subject image by traversing through it
#NOTE: Needs to be made more robest for module mode
def random_snips(image,r, n_snips):
    #Arguments -->
    #image   : 3D array for snipping. (Note: It must be MNI registered.)
    #r       : radius of the snipped image(in pixels)
    #n_snips : no of snips to be generated
    from random import randint
    import numpy as np
    import os
    cur_path = os.getcwd()
    indexes = []
    snips = []
    #import sys, os
    #sys.path.append(os.path.)
    from pfca.core.preprocessing import mni_template_registration
    harvard_mni = mni_template_registration(cur_path,harvard_atlas,'harvard')
    harvard_mask = (harvard_mni.numpy() > 0)*1
    indices = np.asarray(np.where(harvard_mask == 1)).T
    n_indices = len(indices)
    for i in range(n_snips):
        t = randint(0,n_indices)
        tmp_pt = indices[t]
        indexes.append(tmp_pt)
        tmp_snip = image[tmp_pt[0]-r:tmp_pt[0]+r, tmp_pt[1]-r:tmp_pt[1]+r, tmp_pt[2]-r:tmp_pt[2]+r]
        snips.append(tmp_snip)
    return snips, indexes

#Store images as HDF dataset files
#Note: HDF format does not allow a lot of multilabelling...
#This will be overcame with the help of Pandas Dataframe
def store_dataset(images, labels, name):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    import h5py
    import os
    cur_path = os.getcwd()
    num_images = len(images)

    hdf5_dir = cur_path + '/datasets/learning/'
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir + f"{num_images}_" + name + ".h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U16BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U16BE, data=labels
    )
    file.close()
    print("Dataset " + name + " stored Successfully!")

#Read the datasets stored as HDF file format
def read_hdf_dataset(name):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    import os
    import h5py
    cur_path = os.getcwd()
    images, labels = [], []
    hdf5_dir = cur_path + '/datasets/learning/'
    # Open the HDF5 file
    file = h5py.File(hdf5_dir + f"{name}" + ".h5", "r+")
    return file

#creating a filler image for the border pixels so that there are no edges with zero elements
#Not very useful in the augmentation tasks as the amount of empty border pixels are hard to predict in advance
def border_filler(image, border_width):
    x,y,z = image.shape
    tmp_im = image.copy()
    tmp_im[1:(x-border_width), 1:(y-border_width), 1:(z-border_width)] = 0
    return tmp_im

#Rotation and shape deformation for the augmentation task
#Tip : It can be used in a loop to generate rotated images with arbitrary angles
def im_rotate(image, angle, axes):
    #Arguments:
    #image - 3D or 2D image to be rotated
    #angle - should be in Degrees
    border_width = 1
    from scipy import ndimage
    x,y,z = image.shape
    img_rot = ndimage.rotate(image, angle, reshape = False, axes = axes)
    tmp = img_rot.copy()
    tmp[1:(x-border_width), 1:(y-border_width), 1:(z-border_width)] = 0
    without_border = img_rot - tmp
    border = border_filler(image,1)
    img_adj = without_border + border
    return img_adj

#center cropping
#cropping the image in a specified range from the center
#Done in order to eliminate the edge effect on the rotated image(as shown above)
#works only on the 3D image
#TIP : It can be used along with the im_rotate() function for successful augmentation
def center_crop(image, r):
    center = [int(image.shape[0]/2), int(image.shape[1]/2), int(image.shape[2]/2)]
    crop = image[(center[0] - r):(center[0] + r), (center[1] - r):(center[1] + r), (center[2] - r):(center[2] + r)]
    return crop



##################################################################################################################
#MOST IMPORTANT CODE CHUNK IN  THE MODULE
# dictionary generation for data handling
# This dictionary will be easily managable using this function since it will be able to append more data
# into the same dictionary
##################################################################################################################
class data_dictionary:
    def __init__(self):
        import pandas as pd
        import numpy as np
        self.dataset = pd.DataFrame(columns=['image', 'label', 'session'])
        self.splitted = False
        # self.labels = None

    def append(self, data, labelling, session=np.nan, resplit=False):
        # add the images into the dictionary along with the label
        # dataset : list of image patches
        # label : target/non-target as 0/1
        for i in range(len(data)):
            if labelling == 'target':
                temp = {'image': [data[i]], 'label': 1, 'session': session}
                # print(labelling)
            elif labelling == 'non_target':
                temp = {'image': [data[i]], 'label': 0, 'session': session}
                # print(labelling)
            else:
                temp = {'image': [data[i]], 'label': labelling, 'session': session}
            # append the temporary dataframe into the dataset frame
            tempo = pd.DataFrame.from_records(temp)
            self.dataset = self.dataset.append(tempo, ignore_index=True, sort=True)
        # Finally, if some data is appended, it can be splitted again
        if resplit == True:
            self.splitted = False
            print("Respilit allowed. To split the data again, use the train_test_split() now!")

    def disp_by_index(self, index):
        # display a specific image according to the index value of the dataset matrix
        return (self.dataset.iloc[index])['image']

    def return_by_label(self, label):
        # Return a list of images based on the label(target/non-target)
        k = self.dataset
        if label == 'target':
            return k.loc[k['label'] == 1]
        elif label == 'non_target':
            return k.loc[k['label'] == 0]

    def train_test_split(self, ratio=0.7, balanced=True):
        # Arguments:
        # data_dictionary - dictionary datatype which contains labels for the images(0 - non target, 1 - target)
        # ratio - train_dataset_size / total_dataset_size
        #
        # give session labels to the data points : 'train', 'test', 'validation'
        k = self.dataset
        targets = k.loc[k['label'] == 1]
        non_targets = k.loc[k['label'] == 0]

        n_targets = len(targets)
        n_non_t = len(non_targets)

        n_comm = min(n_targets, n_non_t)
        list_target = targets.index.values
        list_non_targets = non_targets.index.values

        if self.splitted == True:
            print("Dataset splitting already happened! Aborting..")
        elif balanced == True and self.splitted == False:  # It will produce a balanced class with equal no of targets and non-targets
            # Decategorizing data in case of appending
            k['session'] = np.nan

            num = int(ratio * n_comm)  # no of target points to select

            trainees = np.random.choice(list_target, size=(num,), replace=False)
            trainees2 = np.random.choice(list_non_targets, size=(num,), replace=False)
            all_train = np.append(trainees, trainees2)
            k.loc[all_train, ['session']] = 'train'  # Labelling the datapoints as training points

            num_test = n_comm - num  # No of test target points
            target_left = k.loc[(k['label'] == 1) & (k['session'] != 'train')]
            non_targets_left = k.loc[(k['label'] == 0) & (k['session'] != 'train')]
            list_target_left = target_left.index.values
            list_non_targets_left = non_targets_left.index.values
            tests1 = np.random.choice(list_target_left, size=(num_test,), replace=False)
            tests2 = np.random.choice(list_non_targets_left, size=(num_test,), replace=False)
            all_test = np.append(tests1, tests2)
            k.loc[all_test, ['session']] = 'test'  # Labelling the datapoints as training points
            self.splitted = True
            self.dataset = k
            return

    def feature_matrix_raw(self, session, ravel=True):
        if self.splitted == False:
            print("Dataset not ready for classification step. Please perform train-test split first.")
        elif self.splitted == True:
            k = self.dataset
            all_session_data = k.loc[k['session'] == session]

            img_dim = (all_session_data.iloc[0]['image']).shape
            feature_vector_len = img_dim[0] * img_dim[1] * img_dim[2]

            features = []
            labels = []

            for idx in range(len(all_session_data)):
                # print(idx)
                t_img = np.array((all_session_data.iloc[idx])['image'])
                labels.append(all_session_data.iloc[idx]['label'])
                features.append(t_img)

            labels = np.array(labels)

            if ravel == True:
                feature_matrix = np.zeros((len(all_session_data), feature_vector_len))
                for i in range(len(all_session_data)):
                    feature_matrix[i, :] = np.ravel(features[i])
                return feature_matrix, labels
            elif ravel == False:
                return features, labels

    def show(self):
        return self.dataset

#######################################################################################################################
#######################################################################################################################

