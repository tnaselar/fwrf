import lasagne
import numpy as np
from theano import tensor as tnsr
from theano import function, shared


##convenience functions for getting/setting params by name

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