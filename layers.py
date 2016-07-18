import lasagne
import numpy as np
from theano import tensor as tnsr
from theano import function, shared


##TODO: GENERALIZE SO NOT COMMITTED TO GAUSSIAN RF MODEL
class receptive_field_layer(lasagne.layers.Layer):
    '''
    receptive_field_layer(incoming, num_voxels, (X_grid, Y_grid), deg_per_stim, x0=None, y0=None, sig=None)
    
    --inputs
    incoming ~ should be an input layer with (T,D,S,S) input_var tensor4
    
    num_voxels  ~ V
    
    (X_grid, Y_grid) ~ numpy matrices of spatial 2D grid specified in pixels. like output of "make_space" or np.meshgrid
    
    when x0,y0,sig not vien initializes each rf with (x0,y0)=(0,0), stdev = 1.
    otherwise, can initialize with a theano shared variable or numpy array
    
    creates a stack of 2D gaussian rf blobs. The stack has dimensions (V,S,S). just call "make_rf_stack()"
    
    application gives a tensor of shape (T,D,V) 
    
    --attributes
    params are [x0, y0, sig], which are the (x,y) coors and standard dev., resp., of the gaussian rfs
    for some set of voxels
    
    make_rf_stack() will construct a stack of visualizable rf's
    '''    
    def __init__(self, incoming, num_voxels, space_grid, deg_per_stim, x0=None, y0=None, sig=None, dtype='float32', **kwargs):
        super(receptive_field_layer, self).__init__(incoming, **kwargs)  ##this will give us an "input_shape" attribute
        self.S = self.input_shape[-1]
        self.V = num_voxels
        
        self.pix_per_deg = self.S * (1./deg_per_stim)  ##this will convert into pixels
        self.Xm = shared(space_grid[0].astype(dtype))  ##explicit casting. is there a better way to do this?
        self.Ym = shared(space_grid[1].astype(dtype))        
        
        if x0 is not None:
            self.x0 = self.add_param(x0,(self.V,), name='x0',rf_param=True,x0=True)
        else:
            self.x0 = self.add_param(np.zeros(self.V,dtype=dtype),(self.V,), name='x0',rf_param=True,x0=True)
        
        if y0 is not None:
            self.y0 = self.add_param(y0,(self.V,), name='y0',rf_param=True,y0=True)
        else:
            self.y0 = self.add_param(np.zeros(self.V,dtype=dtype),(self.V,), name='y0',rf_param=True,y0=True)
        
        if sig is not None:
            self.sig = self.add_param(sig,(self.V,), name='sig',rf_param=True,sig=True) 
        else:
            self.sig = self.add_param(np.ones(self.V,dtype=dtype),(self.V,), name='sig',rf_param=True,sig=True) 
        
        

        
        
        ##we put the pix_per_deg conversion here becuase x0,y0,sig will be shared across feature spaces
        #this expression has shape (V,S,S)
        self.gauss_expr = ((1. / 2*np.pi*(self.sig[:,np.newaxis,np.newaxis]*self.pix_per_deg)**2)
                           *tnsr.exp(-((self.Xm[np.newaxis,:,:]-self.x0[:,np.newaxis,np.newaxis]*self.pix_per_deg)**2
                                       + (self.Ym[np.newaxis,:,:]-self.y0[:, np.newaxis,np.newaxis]*self.pix_per_deg)**2)
                                     /(2*(self.sig[:,np.newaxis,np.newaxis]*self.pix_per_deg)**2)))
        
        self.make_rf_stack = function([],self.gauss_expr)
        
    
    #T,D,S,S x V,S,S --> T,D,V
    def get_output_for(self, input, **kwargs):
        return tnsr.tensordot(input, self.gauss_expr, axes=[[2,3],[1,2]])
    
    def get_output_shape_for(self, input_shape): ##(T,D,V)
        return (input_shape[0], input_shape[1], self.V)
    
    def set_value_for_voxel(self, voxel_idx=None, x0=None, y0=None, sig=None):
        if x0 is not None: ##we assume the shared variable has already been created and registered
            x0_temp = self.x0.get_value()
            x0_temp[voxel_idx] = x0
            self.x0.set_value(x0_temp)
        if y0 is not None:
            y0_temp = self.y0.get_value()
            y0_temp[voxel_idx] = y0
            self.y0.set_value(y0_temp)
        if sig is not None:
            sig_temp = self.sig.get_value()
            sig_temp[voxel_idx] = sig
            self.sig.set_value(sig_temp)
        
    
class compressive_nonlinearity_layer(lasagne.layers.Layer):
    '''
    a compressive function used in a previous publication. works well.
    requires all inputs to be positive, BUT DOES NOT CHECK!
    computes elementwise log(1+sqrt(input))
    '''
    def get_output_for(self, input, **kwargs):
        return tnsr.log(1+tnsr.sqrt(input))
    

class normalization_layer(lasagne.layers.Layer):
    '''
    normalization_layer(incoming,
                        mean=lasagne.init.Constant([0]),
                        stdev=lasagne.init.Constant([1]),
                        mask = lasagne.init.Constant([1]), **kwargs)
    inputs:
        incoming ~ lasagne layer
        mean ~ will be subtracted from input
        stdev ~ divided out of input
        mask ~ a mask applied to input
    outputs:
        does not change input shape
    
    '''
    def __init__(self, incoming, mean=lasagne.init.Constant([0]), stdev=lasagne.init.Constant([1]), mask = lasagne.init.Constant([1]), **kwargs):
        super(normalization_layer,self).__init__(incoming, **kwargs)
        self.mean = self.add_param(mean, self.input_shape[1:], name='mean', trainable=False)
        self.stdev = self.add_param(stdev, self.input_shape[1:], name='stdev', trainable=False)
        self.stability_mask = self.add_param(mask, self.input_shape[1:], name='stability_mask', trainable=False)
    
    def get_output_for(self, input, **kwargs):
        return tnsr.switch(self.stability_mask>0, (input - self.mean[np.newaxis,:,:])/self.stdev[np.newaxis,:,:], 0)

class feature_weights_layer(lasagne.layers.Layer):
    '''
    feature_weights_layer(incoming, NU = lasagne.init.Constant([0]))
       incoming ~ should be an rf space tensor (T,D,V)
             NU ~ (D, V) matrix of feature weights.
    output      ~ (T, V) matrix of predicted responses.
    '''    
    def __init__(self, incoming, NU = lasagne.init.Constant([0]), **kwargs):
        ##this will give us an "input_shape" attribute
        super(feature_weights_layer, self).__init__(incoming, **kwargs)  
        self.D = self.input_shape[1]
        self.V = self.input_shape[-1]
        self.feature_dim = 0
        self.voxel_dim = 1
        ##creates the theano shared variable
        self.NU = self.add_param(NU, (self.D, self.V), name='feature_weights', feature_weights=True) 
    
    def get_output_for(self, input, **kwargs):
        return (input*self.NU[np.newaxis,:,:]).sum(axis=1)
        
    def get_output_shape_for(self, input_shape): ##input_shape = (T, D, V)
        return (input_shape[0], self.V)
    
    ##if NU = None, this will do nothing and return None
    def set_weight_for_voxel(self, voxel_idx=None, NU=None):
        if NU is not None:
            NU_temp = self.NU.get_value()
            NU_temp[:,voxel_idx] = NU
            self.NU.set_value(NU_temp)
            

class prediction_menu_layer(lasagne.layers.Layer):
    '''
    feature_weights_layer(incoming, num_voxels, NU = lasagne.init.Constant([0]))
         incoming ~ should be an rf space tensor (T,D,G), where G is the grid of potential receptive fields
       num_voxels ~ number of voxels to make predictions for. num_voxels = V
               NU ~ (D, G, V) matrix of feature weights. 
    outputs a (T, G, V) menu of predicted responses. V predictions for each of the G potential rf models.
    '''    
    def __init__(self, incoming, num_voxels, NU = lasagne.init.Constant([0]), **kwargs):
        ##this will give us an "input_shape" attribute
        super(prediction_menu_layer, self).__init__(incoming, **kwargs)  
        self.D = self.input_shape[1]
        self.G = self.input_shape[-1]
        self.feature_dim = 0
        self.voxel_dim = 2
        self.V = num_voxels
        ##creates the theano shared variable
        self.NU = self.add_param(NU, (self.D, self.G, self.V), name='feature_weights', feature_weights=True) 
    
    def get_output_for(self, input, **kwargs):
        return (input[:,:,:,np.newaxis]*self.NU[np.newaxis,:,:, :]).sum(axis=1)
        
    def get_output_shape_for(self, input_shape): ##should be (T, G, V)
        return (input_shape[0], self.G, self.V)
        
    ##if NU = None, this will do nothing and return None
    def set_weight_for_voxel(self, voxel_idx=None, NU=None):
        if NU is not None:
            NU_temp = self.NU.get_value()
            NU_temp[:,:,voxel_idx] = NU
            self.NU.set_value(NU_temp)
            