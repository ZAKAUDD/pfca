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