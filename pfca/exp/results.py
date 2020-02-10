#display all the results in a mosaic format in a PDF file

#mosaic plotting
#plotting all the results in a single pdf file
import numpy as np
import pandas as pd
import matplotlib
def results_mosaic(rst_output_matrix, image_array, params):
    #rst_output_matrix : 
    import datetime
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    l = len(rst_output_matrix)
    matrix_size = [(int(l/2) if (l+1)%2 == 0 else int(l/2)+1),2]
    print(matrix_size)
    with PdfPages('multipage_pdf.pdf') as pdf:
        for j in range(image_array.shape[2]):
            plt.figure(figsize = (10,12))
            plt.subplot(matrix_size[0],matrix_size[1],1)
            ax = plt.gca()
            im = ax.imshow(p2_arr[:,:,j], cmap = plt.get_cmap('gray'))
            plt.title("SWI MRI slice of the data")
            plt.colorbar(im, fraction = 0.046, pad = 0.04)
            for i in range(l):
                plt.subplot(matrix_size[0],matrix_size[1],i+1+1)
                ax = plt.gca()
                im = ax.imshow((rst_output_matrix[i])[:,:,j], cmap = plt.get_cmap('gray'))
                plt.title("Hyperparameters -> " + "alpha : " + str((params[i])['alpha']) 
                          + ", beta: " + str((params[i])['beta']) + ", r : " + str((params[i])['r']))
                plt.colorbar(im, fraction = 0.046, pad = 0.04)
                #plt.savefig(cur_path + '/visuals/stills_2d/' + name)
            pdf.savefig()
            plt.close()    


#draw ROI on the images and display the results
def draw_roi(image,peaks,r):
    ##Arguments:
    # image : 3D image on which the labelling is needed to be done
    # peaks : array containing the list of local maximas
    # r     : radius of the ROI bounding box
    import os
    import datetime
    from matplotlib import pyplot as plt
    from skimage.draw import rectangle_perimeter
    import matplotlib.patches as mpathches
    cur_path = os.getcwd()
    name = str((datetime.datetime.now()).strftime("%d%m%Y_%H%M%S")) + '.png'
    n = len(peaks)
    grid_n = int(n/2 if n%2 == 0 else ((n+1)/2))
    
    plt.figure(figsize = (12,14))
    for i in range(n):    
        p = peaks[i]
        temp = image[:,:,p[2]]
        plt.subplot(grid_n,2,i+1)
        plt.imshow(temp, cmap = plt.get_cmap('gray'))
        ax = plt.gca()
        rect = mpathches.Rectangle((p[1]-r, p[0]-r), 2*r, 2*r,
                                  fill= False, edgecolor = 'red', linewidth = 1)
        ax.add_patch(rect)
        #ax.set_axis_off()
    plt.tight_layout() 
    plt.savefig(cur_path + '/visuals/stills_2d/' + name)
    plt.show()           
   
#Snipping out the cubical 3D region out of the candidate detection based on local peaks
def roi_snipping(image,peaks,r):
    import numpy as np
    roi_array = []  #list in which all the roi images will be stored
    for i in range(len(peaks)):
        p = peaks[i]
        temp = image[p[0]-r:p[0]+r, p[1]-r:p[1]+r, p[2]-r:p[2]+r]
        roi_array.append(temp)
    return roi_array


#generate colormap for the labels provided
def generate_colormap(array, label, color, c_matrix = None, l_matrix = None):
    #Arguments:
    #array  : just to get the size of the colormap
    #label  : the string label for all the datapoints in the array
    #color  : BGR value of the color where B,G,R are between 0 to 1(correspondingly 0 to 255)
    #matrix : matrix in which the colormap is needed to be appended
    n = array.shape[0]
    c = (color[0], color[1], color[2])
    if type(c_matrix).__name__ == 'NoneType' and type(l_matrix).__name__ == 'NoneType':
        l_matrix = {}
        l_matrix['0:'+str(n-1)] = label
        c_matrix = []
        for i in range(n):
            c_matrix.append(c)
        return c_matrix,l_matrix    
    elif type(c_matrix).__name__ != 'NoneType' and type(l_matrix).__name__ != 'NoneType':
        ind_list = list(l_matrix.keys())[-1]    #getting the last added label into l_matrix
        last_ind = int(ind_list.split(':')[-1]) #Converting the last index to an integer
        new_ind = last_ind + 1
        l_matrix[str(new_ind) + ':' + str(new_ind + n)] = label
        for i in range(n):
            c_matrix.append(c)
        return c_matrix, l_matrix    


#generate feature vector matrix for the visualization purpose
#NOTE: Works only for single snippet
def feature_matgen(data_pt,matrix = None):
    #Aruments:
    #data_pt : multidimensional data point which is to be added to matrix
    #matrix  : feature_matrix in which data_pt will be appended
    if type(matrix).__name__ == 'NoneType':
        feature_matrix = np.ravel(data_pt)
        feature_matrix = np.reshape(feature_matrix, (1,len(feature_matrix)))
        return feature_matrix
    else:
        temp = np.ravel(data_pt)
        if len(temp) != matrix.shape[1]:
            print("Error: The dimensions of matrix and unravelled data point doesn't match. Appending can't be done.")
        else:
            feature_matrix = np.vstack((matrix,temp))
            return feature_matrix

#converting the whole snippet list into a feature matrix
def snip_matrix(snip_list):
    #Arguments:
    #snip_list : list of snipped images 
    feature_matrix = feature_matgen(snip_list[0])
    for i in range(1, len(snip_list)):
        feature_matrix = feature_matgen(snip_list[i], feature_matrix)
    return feature_matrix


# 2D visualization using t-SNE, Isomap and PCA for estimating the effectiveness of feature space
# NOTE: Dimensions are reduced across columns
def viz_dimensional(feature_matrix, algo='tsne', colormap=None, labelmap=None, txtlabels=False):
    from matplotlib import pyplot as plt
    import datetime
    import os
    cur_path = os.getcwd()
    name = str((datetime.datetime.now()).strftime("%d%m%Y_%H%M%S")) + 'dimReduction.png'
    if algo == 'tsne':
        from sklearn.manifold import TSNE
        x_embed = TSNE(n_components=2).fit_transform(feature_matrix)
        plt_title = 't-Stochastic Neighbour Embedding(t-SNE) plot'
    elif algo == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, svd_solver='full')
        pc_mat = PCA(n_components=40, svd_solver='full').fit(feature_matrix)
        x_embed = pca.fit_transform(feature_matrix)
        plt_title = 'Principal Component Analysis(PCA) plot'
    elif algo == 'isomap':
        from sklearn.manifold import Isomap
        x_embed = Isomap(n_components=2).fit_transform(feature_matrix)
        plt_title = 'Isometric Mapping(Isomap) plot'
    elif algo == 'all':
        return
    else:
        print("Error: Some unknown value of algo argument is encountered. Aborting..")
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    scale = np.ones(x_embed.shape[0]) * 100
    print(x_embed.shape)
    if (colormap == None) and (labelmap == None):
        ax.scatter(x_embed[:, 0], x_embed[:, 1], alpha=0.3, s=scale)
    elif (colormap != None) and (labelmap != None):
        for ind in list(labelmap.keys()):
            index = ind.split(':')
            ax.scatter(x_embed[int(index[0]):int(index[1]), 0], x_embed[int(index[0]):int(index[1]), 1],
                       alpha=0.3, s=scale, c=colormap[int(index[0]):int(index[1])], label=labelmap[ind])
        ax.legend()
    else:
        print("Error: Either colorlist or label list is missing from the arguments.")

    if txtlabels == True:
        n = feature_matrix.shape[0]
        txts = list(np.arange(n) + 1)
        for i, txt in enumerate(txts):
            ax.annotate(txt, (x_embed[i, 0], x_embed[i, 1]))

    ax.grid(True)
    plt.title(plt_title)
    plt.savefig(cur_path + '/visuals/stills_2d/' + name)
    # ax.set_axis_off()
    plt.show()
    if algo == 'pca':
        return x_embed, pc_mat
    else:
        return x_embed
                        
####################################################################################################################
###################################################################################################################
#Heat map function code....Particularly useful to plot Confusion matrix and other parameter results
#Courtesy of Vishwas Mishra who insisted me on developing the heatmap code....

#helper function taken from matplotlib website
#Refer to the following page :
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
################################################################################################################
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    import matplotlib.pyplot as plt
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    import matplotlib.pyplot as plt
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

#######################################################################################################################
#######################################################################################################################
