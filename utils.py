import lasagne
import numpy as np
import pandas as pd
from itertools import product
from theano import tensor as tnsr
from theano import function, shared



##=====================LASAGNE UTILS: convenience functions for getting/setting params by name

def get_model_params_names(l_model):
    '''
    Return list of names of all parameters (shared variables)
    '''
    p_names = [shared_var.name for shared_var in lasagne.layers.get_all_params(l_model)]
    return p_names

def set_named_model_params(l_model,**kwargs):
    '''
    set_named_model_params(lasagne_model, param1=param1_value, ...)
    sets the values of all named params. uses theano shared variable method "set_value()"
    returns nothing, modifies the l_model in-place.
    '''
    for shared_var in lasagne.layers.get_all_params(l_model):
        if shared_var.name in kwargs.keys():
            shared_var.set_value(kwargs[shared_var.name])

def get_named_param_shapes(l_model, *args):
    """
    get_named_param_shapes(l_model, 'param1',...)
    
    returns dictionary of name/shape-tuple pairs
    
    """
    param_shape_dict = {}
    for shared_var in lasagne.layers.get_all_params(l_model):
        if shared_var.name in args:
            try:
                param_shape_dict[shared_var.name] = shared_var.get_value().shape
                print 'parameter %s has shape %s' %(shared_var.name, param_shape_dict[shared_var.name])
            except AttributeError:
                print 'no shape attribute available for %s' %(shared_var.name)
    return param_shape_dict    


def get_named_params(l_model, *args):
    """
    get_named_params(l_model, 'param1',...)
    
    return dictionary of name/theano-shared-variable pairs.
    
    if you modify values of the returned shared variables, you will modify the values
    of the corresponding parameters attached to l_model.
    """
    param_value_dict = {}
    for shared_var in lasagne.layers.get_all_params(l_model):
        if shared_var.name in args:
            param_value_dict[shared_var.name] = shared_var
    return param_value_dict

###===================RF UTILS===========================
##make a table describing the sizes and locations of the receptive fields
def make_rf_table(deg_per_stim,deg_per_radius,spacing,pix_per_stim = None):
    '''
    here is the machinery for setting up grid of rfs
    
    note: requires pandas.
    
    make_rf_table(deg_per_stim,deg_per_radius,spacing,pix_per_stim = None)
    
    deg_per_stim   ~ scalar, determined by experiment
    deg_per_radius ~ (min_rad, max_rad, num_rad) specify the range rf sizes
    spacing        ~ scalar, spacing between rfs in deg
    pix_per_stim   ~ integer, default = None. If defined, add columns to rf_table with rf dimensions in pixels.
    returns
        rf_table   ~ pandas dataframe, each row an rf with columns 'deg_per_radius', 'x_deg','y_deg'
                     all units in deg. relative to origin of feature map = (0,0)
                     If pix_per_stim given, add columns 'pix_per_radius' and 'x_pix', 'y_pix' 
    '''
    n_sizes = deg_per_radius[2]
    rf_radii_deg = np.linspace(deg_per_radius[0],deg_per_radius[1],num=n_sizes,endpoint=True)
    
    corners = compute_grid_corners(deg_per_stim, 0, boundary_condition=0) ##<<puts center of stim at (0,0)
    x_deg,y_deg = construct_placement_grid(corners,spacing)
    
    
    number_of_rfs = x_deg.ravel().size*rf_radii_deg.size
    rf_array = np.zeros((number_of_rfs,3))
    all_rfs = product(rf_radii_deg,np.concatenate((x_deg.ravel()[:,np.newaxis], y_deg.ravel()[:,np.newaxis],),axis=1))
    
    for ii,rf in enumerate(all_rfs):
        rf_array[ii,:] = np.array([rf[0],rf[1][0],rf[1][1]])
    
    rf_table = pd.DataFrame(data=rf_array, columns=['deg_per_radius', 'x_deg', 'y_deg'])
    
    if pix_per_stim:
        scale_factor = lambda row: row*pix_per_stim * (1./deg_per_stim) 
        rf_table['pix_per_radius'] = rf_table['deg_per_radius'].apply(scale_factor)
        rf_table['x_pix'] = rf_table['x_deg'].apply(scale_factor)
        rf_table['y_pix'] = rf_table['y_deg'].apply(scale_factor)
    
    return rf_table

##make grids of points for evaluating functions of space
def make_space(n_pix):
    '''
    make_space(n_pix)
    
    make a grid of pixel locations in a visual field that is n_pix on each side
    
    returns Xm,Ym ~ meshgrid outputs. these are pixel on a 2D visual field with (0,0) at the center
    of the visual field. the upper left-hand corner is (-n_pix/2, -n_pix/2), the lower right-hand corner
    would be (n_pix/2, n_pix/2).
    
    if n_pix is not even, then coordinates range bewteen (-(n_pix-1)/2, (n_pix-1)/2+1)
    '''
    if n_pix % 2 == 0:
        pix_min = -n_pix/2
        pix_max = -pix_min
    else:
        pix_min = -(n_pix-1)/2
        pix_max = -pix_min+1
    
    [Xm, Ym] = np.meshgrid(range(pix_min,pix_max), range(pix_min,pix_max));  
  
    return Xm, Ym


def compute_grid_corners(n_pix, kernel_size,boundary_condition=0):
    '''
    compute_grid_corners(n_pix, kernel_size)
    return corners of placement grid in image of size n_pix given kernel_size (radius)
    and a boundary_condition. boundary_condition >= 0 is the distance of the grid corners from the corners
    of the image in units of kernel_size. So boundary_condition=0 means corners of grid are corners of the 
    image, while boundary_condition = 1 would mean the grid corners are one kernel_size away from the corners
    of the image.
    [left, right, top, bottom]
    '''
    if n_pix % 2 == 0:
        pix_min = -n_pix/2
        pix_max = -pix_min
    else:
        pix_min = -(n_pix-1)/2
        pix_max = -pix_min+1
    
    ks = kernel_size*boundary_condition
    return np.array([pix_min+ks,pix_max-ks,pix_min+ks,pix_max-ks])

def compute_grid_spacing(kernel_size,fraction_of_kernel_size):
    '''
    compute_grid_spacing(kernel_size,fraction_of_kernel_size)
    
    returns integer distance in pixels between each kernel.
    spacing =  int(kernel_size*fraction_of_kernel_size)
   
    '''
    
    gs = int(kernel_size*fraction_of_kernel_size)
    return gs

def construct_placement_grid(grid_corners, grid_spacing):
    '''
    X,Y = construct_placement_grid(grid_corners, grid_spacing)
    given [left,right,top,bottom] corners of the grid, and an integer pixel spacing,
    return a grid of kernel placements.
    '''
    num0 = int((grid_corners[1]-grid_corners[0])/grid_spacing)
    num1= int((grid_corners[3]-grid_corners[2])/grid_spacing)
    X,Y = np.meshgrid(np.linspace(grid_corners[0],grid_corners[1],num=num0,endpoint=True),
		  np.linspace(grid_corners[2],grid_corners[3],num=num1,endpoint=True))
    return X,Y