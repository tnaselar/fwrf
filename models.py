import lasagne
import numpy as np
from theano import tensor as tnsr
from theano import function, shared
from types import GeneratorType
from hrf_fitting.src.features import make_space


##TODO: GENERALIZE SO THAT NOT COMMITTED TO GAUSSIAN RF MODEL
##merges multiple feature map stacks, applies receptive field layer, and whatever else you want.
class rf_model_space(object):
    '''  
       
    rf_model_space( feature_map_dict, deg_per_stim, num_voxels, rf_init={x0:array, y0:array, sig:array})
    
    The foundation for constructing a fwrf model. Creates receptive fields for a set of num_voxel voxels.
    
    inputs:
    feature_map_dict ~ dictionary of (T,D_i,S_i,S_i) feature maps. T can be 1. These values are not stored, they
                       are just used get basic name/dimension information for each kind of feature map.
        deg_per_stim ~ for the rf_layer, so that we can interpret things if deg. of visual angle
          num_voxels ~ the number of voxels
             rf_init ~ optional initial receptive field params for each voxels. 
                       must be a dictionary like {x0:x0_array, y0:y0_array, sig:sig_array}, where len(??_array)=num_voxels
    
    attributes:
        feature_depth
        feature_indices
        feature_resolutions
        D
        deg_per_stim
        num_voxels
        input_var_dict
        rf_layer ~ lasagne layer. good for accessing rf parameters
        rf_model_space ~ lasagne layer
        construct_model_space_tensor(feature_map_dict) ~ method
        normalize(calibration_data_generator, epsilon=10e-6, unstable=10e-10) ~ method
    
    
    merges multiple feature map stacks, applies receptive field layer, creates a dictionary of input tensors
    corresponding to each feature map stack, and creates an "rf_layer" attribute for convenient access to rf
    parameters.
    
    normalization: call it once using training data, it computes the -mean/stdev, adds a normalization layer,
                   and then sets the weights.
    
    does not apply the feature_weights layer. too many different branches after rf application for that to be useful.
    
    
    '''
    
    
    def __init__(self, feature_map_dict, deg_per_stim, num_voxels, rf_init=None):
        
        ##record and store basic properties of the feature maps
        self.feature_depth = {}
        self.feature_indices = {}
        self.feature_resolutions = {}
        idx = 0
        for f_key in feature_map_dict.keys():
            self.feature_depth[f_key] = feature_map_dict[f_key].shape[1]
            self.feature_indices[f_key] = np.arange(idx,idx + self.feature_depth[f_key],step=1)
            idx += self.feature_depth[f_key]
            self.feature_resolutions[f_key] = feature_map_dict[f_key].shape[2]
        
        ##total feature depth
        self.D = np.sum(self.feature_depth.values())
        
        ##rf properties
        self.deg_per_stim = deg_per_stim
        self.num_voxels = num_voxels
        if rf_init is None:
            rf_init = {}
            rf_init['x0'] = np.zeros(num_voxels,dtype='float32')
            rf_init['y0'] = np.zeros(num_voxels,dtype='float32')
            rf_init['sig'] = np.ones(num_voxels,dtype='float32')
        
            
    
        ##construct model space : applies rf's to feature maps
        '''
        each dict value is a (T,D_i,S_i,S_i) stack of feature maps, where i indexes each stack of feature maps
        each model space in the list will have output shape (T,D_i,V)
        will concatenate along axis 1 (= D)
        will share x0,y0,sig (the rf params) across feature spaces
        '''
        rf_space_list = []
        self.input_var_dict = {}
        for f_map_name in feature_map_dict.keys():
            
            ##get S_i x S_i
            input_shape = (None,)+feature_map_dict[f_map_name].shape[1:]

            self.input_var_dict[f_map_name] = tnsr.tensor4(f_map_name)

            ##the rf_layer for the first feature map
            l1 = lasagne.layers.InputLayer(input_shape, input_var = self.input_var_dict[f_map_name], name='input_'+f_map_name)
            
            ##the points in 2D space where rf will be evaluated 
            Xm,Ym = make_space(input_shape[-1])
            
            ##only one feature map stack or the first map stack:
            if not rf_space_list:  
                self.rf_layer = receptive_field_layer(l1,num_voxels,(Xm,Ym),deg_per_stim,x0=rf_init['x0'], y0=rf_init['y0'], sig=rf_init['sig'],name='rf_'+f_map_name)
                rf_space_list.append(self.rf_layer)
            else: ##rf centers/sizes will be shared
                rf_space_list.append(receptive_field_layer(l1,num_voxels,(Xm,Ym),deg_per_stim,
                                                              x0=self.rf_layer.x0,
                                                              y0=self.rf_layer.y0,
                                                              sig=self.rf_layer.sig,
                                                              name='rf_'+f_map_name))
    
        self.rf_model_space = lasagne.layers.ConcatLayer(rf_space_list,axis=1, name='rf_model_space')
        
        self.construct_model_space_tensor = function(self.input_var_dict.values(), lasagne.layers.get_output(self.rf_model_space))
    
    def normalize(self, calibration_data_generator, epsilon=0., unstable=10e-10):
        mst = []
        for cal_map in calibration_data_generator():
            mst.append(self.construct_model_space_tensor(*cal_map.values()))
        mst = np.concatenate(mst,axis=0)
        mn = np.mean(mst,axis=0)
        stdev = np.std(mst, axis=0)+epsilon
        stability_mask = (stdev > unstable).astype('float32')
        self.rf_model_space = normalization_layer(self.rf_model_space, mean=mn, stdev=stdev, mask=stability_mask)
        self.epsilon=epsilon
        self.unstable=unstable
        self.construct_model_space_tensor = function(self.input_var_dict.values(), lasagne.layers.get_output(self.rf_model_space))
        

##a fwrf model
##TODO: A "merge_with" method to merge one fwrf with another so it will be easy to batch up voxels.

class fwrf(rf_model_space):
    
    '''
    fwrf(feature_map_dict, deg_per_stim, num_voxels, rf_init=None, bonus_layers=None)

    Creates a fwrf model for multiple voxels.

    inputs:
    feature_map_dict ~ dictionary of (T,D_i,S_i,S_i) feature maps. T can be 1. These values are not stored, they
                       are just used get basic name/dimension information for each kind of feature map.
        deg_per_stim ~ for the rf_layer, so that we can interpret things if deg. of visual angle
          num_voxels ~ the number of voxels
             rf_init ~ optional initial receptive field params for each voxels. 
                       must be a dictionary like {x0:x0_array, y0:y0_array, sig:sig_array}, where len(??_array)=num_voxels
        bonus_layers ~ a callable stack of layers that accepts an incoming layer. if supplied, the rf_model_space
                       lasagne model is treated as incoming to the bonus layer. a feature_weights_layer is then applied.
                       if not supplied, a feature_weights_layer is added directly to rf_model_space.
    
    attributes:
    predicted_activity(*feature_map_dict.values())

    '''
    
    def __init__(self, *args, **kwargs):
        self.bonus_layers = kwargs.pop('bonus_layers', None)
        NU = kwargs.pop('NU', lasagne.init.Constant([0]))
 
        super(fwrf, self).__init__(*args, **kwargs)  ##this creates the rf_model_space

        if self.bonus_layers is not None:
            self.stack_feature_layers = lambda x: feature_weights_layer(self.bonus_layers(x),NU=NU)
        else:
            self.stack_feature_layers = lambda x: feature_weights_layer(x,NU=NU)
            
        self.fwrf = self.stack_feature_layers(self.rf_model_space)
        
        self.pred_expr = lasagne.layers.get_output(self.fwrf)
        
        ##this gives predicted output as (T,V) tensor
        self.pred_func = function(self.input_var_dict.values(), self.pred_expr)
    
    def predict( self, feature_map_dict):
        return self.pred_func(*feature_map_dict.values())

    
    def train_me(self, *args, **kwargs):
        '''
        you can call:
        train_me(trn_data_generator, val_data_generator, fine=True, **training_keywords,)
        or:
        train_me(trn_data_generator, val_data_generator, coarse=True, rf_grid=required_dict_of_rf_params,
                                                                      rf_batch_size=optional,
                                                                      **training_keywords)
        
        note that trn/val_data_generator should be functions that return generataors of batches of data.
        
        so the first batch of training data would be like: first_trn_batch = trn_data_generator().next
        first_trn_batch = (inp, outp)
        inp = feature_map_dict
        outp = matrix of voxel activities with shape (first_batch_size, num_voxels)
        
        optional training_keywords are (with their default values):
            epochs = 1, 
            check_every=10,
            num_iters=10,
            learning_rate = 1.0,
            learn_these_params = None,
            voxel_dims = None,
            model_dims = None,
            check_dims = True,
            print_stuff=False
        
        modifies the feature_weights and the rf parameters in place.
        returns:
        
        fine:    (final_val_loss, improvement_history, trn_loss_history)
        coarse: final_val_loss (sorry, batching the rf's makes it a pita to return full history)
        '''
        
        coarse = kwargs.pop('coarse', False)
        
        if coarse:
            assert kwargs.pop('fine', False) == False, 'you have to decide on coarse or fine fitting. fine is default'
            assert 'rf_grid' in kwargs.keys(), 'coarse traning requires an rf_grid'
            assert type(kwargs['rf_grid']) is dict, 'rf_grid should be a dictionary with keys x0,y0,sig' 
            return self._train_coarse(*args, **kwargs)
                        
        else:
            assert 'rf_grid' not in kwargs.keys(), "don't pass an rf grid if you are fine-tuning. it's confusing."
            fine = kwargs.pop('fine',True)
            return self._train_fine(*args,**kwargs)
            
    
    def _train_fine(self, trn_data_generator, val_data_generator, **kwargs):
        self.fwrf,val_loss,val_history,trn_history = batch_model_learner(self.fwrf, self.input_var_dict,
                                                                         trn_data_generator, val_data_generator,
                                                                         self.num_voxels, **kwargs).learn()
        return val_loss, val_history, trn_history

    def _train_coarse(self, trn_data_generator, val_data_generator, **kwargs):
        
        rf_grid = kwargs.pop('rf_grid','None')
        assert rf_grid is not None, 'you somehow called _train_coarse without passing rf_grid as keyword. are you a wizard?'
        rf_batch_size = kwargs.pop('rf_batch_size',None)
        consolidate = kwargs.pop('consolidate', False)
        normalize = kwargs.pop('normalize', False)
        
  
        if rf_batch_size is None:
            rf_grid_batches = (rf_grid,)
        else: ##break the rf_grid up into batches
            rf_grid_batches = self._batch_rf_grid(rf_grid, rf_batch_size)  ##give us a generator

        ##initialize containers to collect training results across batches of rf's
        best_loss = []     ##collects 1 x V arrays
        best_params = []   ##collects 1 x D x V arrays
        best_rfs = []      ##collects 1 x 3 x V arrays, (x0,y0,sig)
        
        for rfb in rf_grid_batches:
                  
            ##a generator of model space tensors that iterates over T
            trn_gen_multi,val_gen_multi = self._build_rf_model_space_tensor(trn_data_generator,
                                                                            val_data_generator,
                                                                            rfb,
                                                                            normalize=normalize,
                                                                            consolidate=consolidate)
            assert type(trn_gen_multi()) is GeneratorType, 'you somehow managed to not produce a generator'
            assert type(val_gen_multi()) is GeneratorType, 'you somehow managed to not produce a generator'
            

            ##a new network that treats everything up to the feature weights as input
            ##NOTE: IT IS REALLY BAD TO BE REFERENCING A DICT. KEY HERE. SUPER NOT GENERAL!
            G = len(rfb['x0']) ##grab the number of candidate rf models
            proxy_net, proxy_input_var_dict = self._build_proxy_network((None,self.fwrf.D,G))

            ##initialize a learner for the best feature weights for the rf models in rfb
            learn_best_rf = batch_model_learner_multi(proxy_net,
                                                      proxy_input_var_dict,
                                                      trn_gen_multi,
                                                      val_gen_multi,
                                                      self.num_voxels,
                                                      G,
                                                      **kwargs)
            ##do the learning
            _,val_loss,val_history,trn_history = learn_best_rf.learn()

            ##get the index of the best set of weights
            cur_best_loss,best_dx = learn_best_rf.get_min_loss(val_loss)
            best_loss.append(cur_best_loss)
                        
            ##get the best feature weights
            best_params.append(learn_best_rf.select_best_model(val_loss)['feature_weights'])
                       
            ##expose the best rf params for each voxel
            best_rfs.append(self._index_rf_grid_dict(rfb, best_dx))

        best_loss = np.vstack(best_loss)  ##batch x voxels
        best_params = np.stack(best_params, axis=0)   ##batch x D x voxels
        best_rfs = np.stack(best_rfs, axis=0)         ##batch x 3 x voxels
        
        best_batches = tuple(np.argmin(best_loss, axis=0))
        vox_range = tuple(range(self.num_voxels))
        best_params = best_params[best_batches,:,vox_range].T  ##(D,voxels). transpose to put voxels in last dim.
        best_rfs = best_rfs[best_batches,:,vox_range].T ##(3,voxels)
        best_loss = np.min(best_loss,axis=0)
        
         
        ##assign this back to fwrf
        ##BAD: THIS IS SICKENINGLY NOT GENERAL.
        set_named_model_params(self.fwrf, feature_weights=best_params)
        set_named_model_params(self.fwrf, x0=best_rfs[0])
        set_named_model_params(self.fwrf, y0=best_rfs[1])
        set_named_model_params(self.fwrf, sig=best_rfs[2])
        
        ##return training history
        return best_loss
        
        
    def _batch_rf_grid(self, rf_grid,rf_batch_size):
        '''
        returns a proper generator of batches of rf_grid
        '''
        x0 = rf_grid['x0']
        y0 = rf_grid['y0']
        sig = rf_grid['sig']
        n = rf_batch_size
        for ii in xrange(0, len(x0), rf_batch_size):
            yield {'x0':x0[ii:ii+n],'y0':y0[ii:ii+n], 'sig':sig[ii:ii+n]}
    
    def _index_rf_grid_dict(self, rf_grid_dict, dx):
        rf = np.zeros((3,len(dx)))
        rf[0,:] = rf_grid_dict['x0'][dx]
        rf[1,:] = rf_grid_dict['y0'][dx]
        rf[2,:] = rf_grid_dict['sig'][dx]
        return rf
    
    def _build_proxy_network(self, input_shape):
        #check fwrf, get shape
        #make a new network with an input layer and prediction_menu layer only
        proxy_input_var_dict = {}
        proxy_input_var_dict['mst'] = tnsr.tensor3('mst')
        proxy_net = lasagne.layers.InputLayer((None,)+input_shape[1:], input_var = proxy_input_var_dict['mst'])
        proxy_net = prediction_menu_layer(proxy_net, self.num_voxels)
        return proxy_net, proxy_input_var_dict
        
    
    def _build_rf_model_space_tensor(self, trn_data_gen, val_data_gen, rf_grid, normalize=False, consolidate=False):
        '''
        generating_function = _build_rf_model_space_tensor(trn_data_gen, val_data_gen, rf_grid, consolidate=False)
        the goal is to convert a data generator of (feature_map_dict,voxel_activity) tuples
        into a generator of (model_space_tensor, voxel_activity) tuples.
        
        the model_space_tensors will have shape (batch_size,D,G)
        
        returns  *functions* that, when called with no arguments, are each a generator of such tuples.
        so
        
        (first_mst_batch, first_voxel_act_batch) = generating_function().next()
        
        note that, to comply with idea that network inputs should be dictionaries,
        
        first_mst_batch = {'mst':mst}, where mst is a model space tensoro with shape (batch_size,D,G)
        
        '''
        example_inp,_ = trn_data_gen().next()
        G = len(rf_grid['x0']) ##grab the number of candidate rf models
        proxy_rf_model_space = rf_model_space(example_inp, self.deg_per_stim, G,rf_init=rf_grid)
        if normalize:
            ##need a new generator that only generates inputs
            def only_input_trn_gen():
                for inp,outp in trn_data_gen():
                    yield inp
            ##get normalization params       
            proxy_rf_model_space.normalize(only_input_trn_gen)
            
        ##okay, this is ugly but,
        if not consolidate:
            ##generating function for trn data
            def make_trn_gen_func():
                for inp,outp in trn_data_gen():
                    yield ({'mst':proxy_rf_model_space.construct_model_space_tensor(*inp.values())}, outp)
            ##generating function for val data
            def make_val_gen_func():
                for inp,outp in val_data_gen():
                    yield ({'mst':proxy_rf_model_space.construct_model_space_tensor(*inp.values())}, outp)
                    
        else:
            ##we don't want batches, so we smoosh them all together
            def make_trn_gen_func():
                mst = []
                r = []
                for inp,outp in trn_data_gen():
                    r.append(outp)
                    mst.append(proxy_rf_model_space.construct_model_space_tensor(*inp.values()))
                mst = np.concatenate(mst, axis=0)
                r = np.concatenate(r, axis=0)
                yield ({'mst':mst},r)

            def make_val_gen_func():
                mst = []
                r = []
                for inp,outp in val_data_gen():
                    r.append(outp)
                    mst.append(proxy_rf_model_space.construct_model_space_tensor(*inp.values()))
                mst = np.concatenate(mst, axis=0)
                r = np.concatenate(r, axis=0)
                yield ({'mst':mst},r)

        return make_trn_gen_func, make_val_gen_func
    
    def normalize(self, *args, **kwargs):
        ##TODO: Should check to make sure no normalization yet!
        
        ##call the model_space method
        super(fwrf, self).normalize(*args, **kwargs)
        
        ##then overwrite the feature layer
        self.fwrf = self.stack_feature_layers(self.rf_model_space)
        
        ##and update the prediction expressions/functions
        self.pred_expr = lasagne.layers.get_output(self.fwrf)
        self.pred_func = function(self.input_var_dict.values(), self.pred_expr)
        
               
    
