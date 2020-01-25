import numpy as np
#3d version of radial symmetry transform algorithm
def rst_3d(image, radii, alpha, beta):
    from scipy.ndimage import sobel, generic_gradient_magnitude, gaussian_filter
    from time import time
    #initiating the output image array
    t0 = time()

    output = np.zeros(image.shape)
    workingDims = tuple((e) for e in image.shape)
    
    O_n = np.zeros(workingDims, np.int16)
    M_n = np.zeros(workingDims, np.int16)    
    
    #calculating the gradients in all directions and the magnitude image
    grad_image = generic_gradient_magnitude(image, sobel)
    gx = sobel(image,0)
    gy = sobel(image,1)
    gz = sobel(image,2)
    
    #cutoff beta for removing some of the smaller gradients
    gthres = np.amax(grad_image)*beta
    
    #calculating negatively affected pixels
    gpx = np.multiply(np.divide(gx, grad_image, out=np.zeros(gx.shape), where=grad_image!=0), radii).round().astype(int)
    gpy = np.multiply(np.divide(gy, grad_image, out=np.zeros(gy.shape), where=grad_image!=0), radii).round().astype(int)
    gpz = np.multiply(np.divide(gz, grad_image, out=np.zeros(gz.shape), where=grad_image!=0), radii).round().astype(int)
    
    for coords, gnorm in np.ndenumerate(grad_image):
        if gnorm > gthres:
            i, j, k = coords
            pnve = (i-gpx[i,j,k], j-gpy[i,j,k], k-gpz[i,j,k])
            O_n[pnve] -= 1
            M_n[pnve] -= gnorm
    
    O_n = np.abs(O_n)
    O_n = O_n/float(np.amax(O_n))
    
    M_max = float(np.amax(np.abs(M_n)))
    M_n = M_n/M_max
    
    F_n = np.multiply(np.power(O_n,alpha), M_n)
    
    s = gaussian_filter(F_n, 0.5*radii)
    t1 = time()
    print("Time taken for 3D RST:", t1-t0)

    return s


# Training SVM using the bag of words features
#classifier must be function from Sklearn library or works with similar commands as Sklearn classifiers...
def train_classifier(dataset, feature_column, classifier):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    ###########Data Acquisition##########################
    data = dataset.loc[dataset['session'] == 'train']
    indices = data.index.values
    tmp0 = data[feature_column]
    tmp = tmp0[0]
    X = np.zeros((1, len(tmp)))
    for idx in (indices):
        tmp1 = (dataset[dataset['session'] == 'train'])[feature_column]
        tmp2 = np.array(tmp1[idx])
        X = np.vstack((X, tmp2))
    X = np.delete(X, 0, 0)
    # print(X.shape)
    # Why it is so hard to put arrays in there
    # X = data[feature_column]
    Y = data['label'].to_numpy()
    Y = Y.astype('int')
    # print(Y.shape)
    #######################################################
    # sys.exit()
    classifier.fit(X, Y)
    y_pred = classifier.predict(X)
    tn, fp, fn, tp = (confusion_matrix(Y, y_pred)).ravel()

    confusion_mat = np.array([[tp, fp],
                              [fn, tn]])
    fig, ax = plt.subplots()

    im, cbar = heatmap(confusion_mat, ['Predicted True', 'Predicted False'], ['Actual True', 'Actual False'], ax=ax,
                       cmap="Blues", cbarlabel=" [No. of samples]")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    fig.tight_layout()
    plt.show()
    print("Sensitivity : ", (tp / (fn + tp)))
    print("Specificity : ", (fp / (fp + tn)))
    return classifier