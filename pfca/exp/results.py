#display all the results in a mosaic format in a PDF file

#mosaic plotting
#plotting all the results in a single pdf file
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
    

     