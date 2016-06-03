import numpy as np
import lasagne
import theano.tensor as tnsr
from theano import function,shared
​

####----static version

class receptive_field_layer(lasagne.layers.Layer):

    '''

    shape of input ~ (T,D,S,S)

    rf_stack ~ shape = (G,S,S)

    '''    

    def __init__(self, incoming, num_rfs, rf_stack, **kwargs):

        super(receptive_field_layer, self).__init__(incoming, **kwargs)  ##this will give us an "input_shape" attribute

        self.S = self.input_shape[-1]

        self.G = num_rfs

        self.rf_stack = self.add_param(rf_stack, (self.G,self.S,self.S), name='rf_stack') ##creates the theano shared variable

    

    def get_output_for(self, input, **kwargs):

        return tnsr.tensordot(self.rf_stack, input, axes=[[1,2],[2,3]])

    

    def get_output_shape_for(self, input_shape): ##(G, T, D)

        return (self.G, input_shape[0], input_shape[1])


####----parameterized version
class receptive_field_layer(lasagne.layers.Layer):
    '''
    shape of input ~ (T,D,S,S)
    rf_stack ~ shape = (G,S,S)
    '''    
    def __init__(self, incoming, num_rfs, space_grid, **kwargs):
        super(receptive_field_layer, self).__init__(incoming, **kwargs)  ##this will give us an "input_shape" attribute
        self.S = self.input_shape[-1]
        self.G = num_rfs
        self.Xm = shared(space_grid[0])
        self.Ym = shared(space_grid[1])
        self.x0 = self.add_param(np.zeros(num_rfs), (num_rfs,), name='x0')
        self.y0 = self.add_param(np.zeros(num_rfs), (num_rfs,), name='y0')
        self.sig = self.add_param(np.ones(num_rfs), (num_rfs,), name='sig')
        self.gauss_expr = ((1. / 2*np.pi*self.sig[:,np.newaxis,np.newaxis])
                           *tnsr.exp(-((self.Xm[np.newaxis,:,:]-self.x0[:,np.newaxis,np.newaxis])**2 + (self.Ym[np.newaxis,:,:]-self.y0[:, np.newaxis,np.newaxis])**2)
                                     /(2*self.sig[:,np.newaxis,np.newaxis]**2)))
        
        self.make_rf_stack = function([],self.gauss_expr)
        
    
    def get_output_for(self, input, **kwargs):
        return tnsr.tensordot(self.gauss_expr, input, axes=[[1,2],[2,3]])
    
    def get_output_shape_for(self, input_shape): ##(G, T, D)
        return (self.G, input_shape[0], input_shape[1])