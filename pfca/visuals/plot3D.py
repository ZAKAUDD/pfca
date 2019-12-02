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


#3d plotting and storing the array in vtk folder
#data = p2_arr
def plot3d(data = None, patient_name = 'unknown', file = None):
    from mayavi import mlab
    import datetime
    string = str((datetime.datetime.now()).strftime("%d%m%Y_%H%M%S"))
    if type(file).__name__ == 'NoneType' and type(data).__name__ != 'NoneType':
        np.save(cur_path + '/visuals/3d_vtk/' + patient_name +  '_' + string + '.npy', data)
    elif type(file).__name__ != 'NoneType' and type(data).__name__ == 'NoneType':
        data = np.load(cur_path + '/visuals/3d_vtk/' + file)

    #3d plotting
    from mayavi import mlab
    mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))
    src = mlab.pipeline.scalar_field(data)
    voi = mlab.pipeline.extract_grid(src)
    mlab.pipeline.iso_surface(voi, colormap='Spectral')
    mlab.show()


#3d plotting with data overlayed on the template provided
#template brain needed to be provided or must be present in the storage folder
def plot_brain(template):
    from mayavi import mlab
    import datetime
    string = str((datetime.datetime.now()).strftime("%d%m%Y_%H%M%S"))
    
    #3d plotting
    from mayavi import mlab
    mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))
    src = mlab.pipeline.scalar_field(template)
    voi = mlab.pipeline.extract_grid(src)
    mlab.pipeline.iso_surface(voi, colormap='Spectral')

    #src1 = mlab.pipeline.scalar_field(data)
    #voi1 = mlab.pipeline.extract_grid(src1)
    #mlab.pipeline.iso_surface(voi1, colormap='Spectral')
    mlab.show()


#3d plotting with data overlayed on the template provided
#template brain needed to be provided or must be present in the storage folder
def plot_bleeds3d(data, template):
    from mayavi import mlab
    import datetime
    from tvtk.util.ctf import PiecewiseFunction, ColorTransferFunction
    string = str((datetime.datetime.now()).strftime("%d%m%Y_%H%M%S"))
    otf = PiecewiseFunction()
    otf.add_point(0.1,0.0)  #opacity values for the boundary elements of the brain...For opaque objects, opacity = 1.0
    otf.add_point(5,0.1)
    otf.add_point(100,0.25)
    
    ctf = ColorTransferFunction()
    ctf.add_rgb_point(0, 0, 0, 0)
    ctf.add_rgb_point(1000, 0.6,1,1)
    #3d plotting
    mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))
    src = mlab.pipeline.scalar_field(template)
    #voi = mlab.pipeline.extract_grid(src)
    vol = mlab.pipeline.volume(src)
    vol._otf = otf
    vol._volume_property.set_gradient_opacity(otf)
    
    vol._volume_property.set_color(ctf)
    vol._ctf = ctf
    vol.update_ctf = True
    
    src1 = mlab.pipeline.scalar_field(data)
    voi1 = mlab.pipeline.extract_grid(src1)
    mlab.pipeline.iso_surface(voi1, color=(1,0,0))
    mlab.show()


            