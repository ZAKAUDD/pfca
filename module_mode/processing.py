import numpy as np
#3d version of radial symmetry transform algorithm
def rst_3d(image, radii, alpha, beta):
    from scipy.ndimage import sobel, generic_gradient_magnitude, gaussian_filter
    #initiating the output image array
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
    
    return s