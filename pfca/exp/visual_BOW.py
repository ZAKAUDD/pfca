
#Extract features using whitening and tiling for visual Bag of Words features
def feature_extraction(image, stride=2, size=2):
    from sklearn.decomposition import PCA
    tiles = img_keypoints(image, stride,size)
    feat = snip_matrix(tiles)
    whitened_data = PCA(whiten = True).fit_transform(feat)
    #normalized_image = normalize(image)
    #whitened_image = whitening(normalized_image)
    #tiles = img_keypoints(whitened_image, stride,size)
    return whitened_data

#Trying out the same procedure on the small patches of each 3D image patches
#taking a (2*2*2) patches in each image with a stride of 2
#So, in a (10*10*10) image, it gives 125 patches or keypoints.

#Generating the 125 patches from each image
#Function works for 3d images and generate 3d patches
def img_keypoints(image, size, stride):
    patches = []
    for i in range(0,image.shape[0],stride):
        for j in range(0,image.shape[1],stride):
            for k in range(0,image.shape[2],stride):
                temp = image[i:i+size, j:j+size, k:k+size]
                #print(temp)
                patches.append(temp)
    return patches

#This function is for collecting the keypoint descriptors from the whole batch
def img_batch_keypoints(batch, size, stride):
    patches = []
    for image in batch:
        for i in range(0,image.shape[0],stride):
            for j in range(0,image.shape[1],stride):
                for k in range(0,image.shape[2],stride):
                    temp = image[i:i+size, j:j+size, k:k+size]
                    #print(temp)
                    patches.append(temp)
    return patches

#Adding visual BOW feature descriptors to the data_dictionary class derived from Pandas DataFrame
def add_features(dataset, session = 'train'):
    data = (dataset.show()).copy()
    session_set = data.loc[data['session'] == session]
    images = session_set['image']
    indices = images.index.values
    tmp_column = pd.DataFrame(columns = ['features'])
    flag = 0
    feat_mat = []
    for idx in indices:
        image = data.iloc[idx]['image']
        #img = np.array(image)
        feat_list = feature_extraction(image)
        temp = {'features': [feat_list]}
        temp_col = pd.DataFrame.from_records(temp)
        tmp_column = tmp_column.append(temp_col, ignore_index = True)
    #print(tmp_column)
    tmp_column['new_index'] = pd.Series(indices, index = tmp_column.index)
    tmp_column = tmp_column.set_index('new_index')
    data.loc[indices, 'features'] = tmp_column['features']
    #dataset.dataset
    #print(tmp_column)
    return data


#Super-features ..... by using k_means clustering
#n_clusters = no of classes * 10s
def quantizer_KMeans(dataset_with_features, session = 'train', n = 20):
    ###############################################################
    ###This initial part deals with the data acquisition#$$$$$$$$$$
    dataset = dataset_with_features
    indices = dataset.loc[dataset['session'] == session].index.values
    tmp0 = (((dataset.loc[dataset['session'] == session]))['features'])
    tmp = tmp0[0]
    features_pool = np.zeros((1,tmp.shape[1]))
    for idx in (indices):
        tmp1 = (datas[datas['session'] == session])['features']
        tmp2 = tmp1[idx]
        features_pool = np.vstack((features_pool,tmp2))
    features_pool = np.delete(features_pool,0,0)
    ##############################################################
    ##############################################################
    #n = 10*2
    #No of clusters(acc to heuristics) ---> no of classes*10
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = n).fit(features_pool)
    return kmeans

def visual_BOW(dataset_with_features, quantizer, session, n = 20):
    ############data acquisition######################
    data = dataset_with_features
    session_set = data.loc[data['session'] == session]
    images = session_set['image']
    indices = session_set.index.values
    tmp_column = pd.DataFrame(columns = ['visual_BOW'])
    for idx in indices:
        histo = np.zeros(n)
        features = np.array(data.iloc[idx]['features'])
        n_keypoints = np.size(features)
        vals = quantizer.predict(features)
        histo[vals] += (1/n_keypoints)
        temp = {'visual_BOW': [histo]}
        temp_col = pd.DataFrame.from_records(temp)
        tmp_column = tmp_column.append(temp_col, ignore_index = True)
    #print(tmp_column)
    tmp_column['new_index'] = pd.Series(indices, index = tmp_column.index)
    tmp_column = tmp_column.set_index('new_index')
    data.loc[indices, 'visual_BOW'] = tmp_column['visual_BOW']


