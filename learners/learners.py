import lasagne
import numpy as np
from theano import tensor as tnsr
from theano import function, shared


class batch_model_learner(object):
    
    def __init__(self,l_model, model_input_tnsr_dict, trn_data_generator, val_data_generator, num_voxels,
                 epochs = 1,
                 check_every=10,
                 num_iters=10,
                 learning_rate = 1.0,
                 learn_these_params = None,
                 voxel_dims = None,
                 check_dims = True,
                 print_stuff=False):

        '''
        batch_model_learner(l_model,
                            model_input_tnsr_dict,
                            trn_data_generator,
                            val_data_generator,
                            num_voxels,
                            epochs=1,
                            check_every=10,
                            num_iters=10,
                            learning_rate = 1.0,
                            learn_these_params = None,
                            voxel_dims = None
                            check_dims = True)

        a class for cross-validated (stochastic) gradient descent on independent encoding models for distinct voxels.
        Uses cross-validation to decide when to stop updating.

        inputs:
                            l_model ~ a lasagne model with output shape (T,V), where T = #trials, V=#voxels
            model_input_tensor_dict ~ dict of theano tensors that are input to the l_model.
                 trn_data_generator ~ a generator of (input, output) training data batches. you must write it.
                                      it should yield batches of data, i.e., trn_data_generator() returns your training batches.
                                      the format for each batch is (input, output), where
                                      input = training input data dictionary.
                                      keys/values match the names/dimensions of tensors in model_input_tensor_dict
                                      output = np.array of shape (Ttrn, V)
                 val_data_generator ~ generator for validation data
                             epochs ~ number of times through all data in the generator
                        check_every ~ int. how often validation loss is checked. default = 10
                          num_iters ~ number of gradient steps per batch before stopping. default = 100
                      learning_rate ~ size of gradient step. default = 1
                 learn_these_params ~ list of parameters names to learn, in case you don't want to train them all.
                                      default = None, meaning train all trainable params
                        voxel_dims  ~ dictionary, keys=param names, values = ints.
                                      for each learned parameter, this corresponds to the voxel dimension.
                                      default = None, in which case we try to figure out what the dimension is.
                                      if we can't, we complain, and you are forced to specify.
                         check_dims ~ if True, run potentially slow sanity check on dimensions of inputs
        outputs:
                     l_model ~ original l_model with params all trained up. this is a convenience, as params are learned in-place
              final_val_loss ~ the final validation loss for each of the models
                 trn_history ~ array of length num_iters showing number of voxels with decreased training loss at each time step
                 val_history ~ array of length num_iters showing number of voxels with decreased validat'n loss at each time step

        notes:
            we call each independent model a "voxel". but it could be anything. 

            the data_generator approach is flexible. in the case where you can simply load up the whole training/validation
            data sets at once, it is just a slight encumberance, forcing you to define trn_data_generator() = yield trn_data.
            in many cases though, it will not be possible to load all the data at once, so this approach will be helpful.

            if trn_data_generator() contains only one large batch, epochs=1, and num_iters = BIG, we get standard grad. descent.
            if trn_data_generator() contains many mini-batches, epochs > 1, num_iters = 1, we get *stochastic* grad. descent.
        '''
        
        ##record all the inputs
        self.l_model = l_model
        self.learn_these_params = learn_these_params
        self.model_input_tnsr_dict = model_input_tnsr_dict
        self.trn_data_generator = trn_data_generator
        self.val_data_generator = val_data_generator
        self.num_voxels = num_voxels
        self.epochs = epochs
        self.check_every=check_every
        self.num_iters = num_iters
        self.learn_these_params = learn_these_params
        self.learning_rate = learning_rate
        self.print_stuff = print_stuff
        self.voxel_dims = voxel_dims
        self.check_dims = check_dims

        ##inspect one batch of input to sanity check dimensions
        if self.check_dims:
            self.grab_example_batch()
            ##check data dimensions
            self.check_consistent()
    
        ##get learned params: stores as a list of shared variables just like output of lasage.layers.get_all_params
        self.get_learned_params()


        ##try to determine which dimension of each parameter is the voxel dimension
        self.find_batch_dimension()

        ##construct a gradient update and loss functions to be called iteratively by the learn method
        self.construct_training_kernel()
    
    def grab_example_batch( self ):
        self.trn_in, self.trn_out = next(self.trn_data_generator()) 
        self.val_in, self.val_out = next(self.val_data_generator())


    def check_consistent( self ):
        ##read first batches to get some dimensions. assumes first dimension of input is time, last dimension is voxel 
        trn_batch_size, num_trn_voxels = self.trn_out.shape[0],self.trn_out.shape[-1]
        val_batch_size, num_val_voxels = self.val_out.shape[0],self.val_out.shape[-1]   
        assert num_trn_voxels == num_val_voxels, "number of trn/val voxels don't match"
        assert num_trn_voxels == self.num_voxels,"number voxels in trn/val does not match specific number of voxels"
        
        
        ##check to make sure that all feature map stacks in the input dictionary have the same number of timepoints.
        ##this is not a complete check, since it's just the first batch, but if shit is not fucked up for this batch, 
        ##shit will hopefully not be fucked up for subsequent batches
        for k,v in self.trn_in.iteritems():
            assert v.shape[0] == trn_batch_size, "number of input/output trn trials don't match for feature %s" %(k)

        for k,v in self.val_in.iteritems():
            assert v.shape[0] == val_batch_size, "number of input/output val trials don't match for feature %s" %(k)


    def get_learned_params( self ):
        if self.learn_these_params is not None:            
            self.params = [v for v in get_named_params(self.l_model, *self.learn_these_params).values()] ##<<unpack the dict.
        else:
            self.params = lasagne.layers.get_all_params(self.l_model,trainable=True)
        print 'will solve for: %s' %(self.params)

    def find_batch_dimension( self ):
        if self.voxel_dims is None:
            self.voxel_dims = {}
            for p in self.params:
                ##the voxel dimension should be the one that matches "num_voxels"
                vdim = [ii for ii,pdim in enumerate(p.shape.eval()) if pdim==self.num_voxels]

                ##if we happen to have multiple dimensions that = "num_voxels", user must disambiguate
                assert len(vdim)==1, "can't determine voxel dimension for param %s. supply 'voxel_dims' argument" %p.name
                self.voxel_dims[p.name] = vdim[0]

                
    ##construct a gradient update and loss functions to be called iteratively by the learn method
    def construct_training_kernel( self ):
        voxel_data_tnsr = tnsr.matrix('voxel_data_tnsr')  ##voxel data tensor: (T x V)

        ##get symbolic prediction expression
        pred_expr = lasagne.layers.get_output(self.l_model)  ##voxel prediction tensor: (T x V)

        ##generate symbolic loss expression
        trn_diff = voxel_data_tnsr-pred_expr        ##difference tensor: shape = (T, V)
        loss_expr = (trn_diff*trn_diff).sum(axis=0) ##sum squared diffs over time: shape = (V,)

        ##for *training* error we compute of errors along voxel dimension.
        ##we have to do this because auto-diff requires. a scalar loss function.
        ##BUT: this is fine because gradient w.r.t. one voxel's weights is not affected by loss for any other voxel.
        trn_loss_expr = loss_expr.sum()

        #construct update rule using *training* loss.
        fwrf_update = lasagne.updates.sgd(trn_loss_expr,self.params,learning_rate=self.learning_rate)
        self.trn_kernel = function([voxel_data_tnsr]+self.model_input_tnsr_dict.values(),
                                   trn_loss_expr,
                                   updates=fwrf_update)           
        print 'will update wrt: %s' % (self.params,)
        
        ##compile loss and training functions
        ##NOTE: this is *validation* loss, not summed over voxels, so it has len = num_voxels
        print 'compiling...'
        self.loss = function([voxel_data_tnsr]+self.model_input_tnsr_dict.values(), loss_expr)
                
    
    ##if the last gradient step made things better for some voxels, update their parameters
    def update_best_param_values(self, best_param_values, improved_voxels):
        for ii,p in enumerate(self.params):
            vdim = self.voxel_dims[p.name] ##the voxel dimension
            s = [slice(None),]*p.ndim      ##create a slicing object with right number of dims
            s[vdim] = improved_voxels      ##assign improved voxel indices to the correct dim of the slice object
            best_param_values[ii][s] = np.copy(p.get_value()[s])   ##keep a record of the best params.
        return best_param_values        
    
    ##iteratively perform gradient descent
    def learn(self):
        
        ##initialize best parameters to whatever they are
        best_param_values = [np.copy(p.get_value()) for p in self.params]
        
        ##initalize validation loss to whatever you get from initial weigths
        val_loss_was = 0
        for val_in, val_out in self.val_data_generator():
            val_loss_was += self.loss(val_out, *val_in.values())

        ##initialize train loss to whatever, we only report the difference    
        trn_loss_was = 0.0 ##we keep track of total across voxels as sanity check, since it *must* decrease
  
        val_history = []
        trn_history = []
        
        ##descend and validate
        epoch_count = 0
        while epoch_count < self.epochs:
            print '=======epoch: %d' %(epoch_count) 
            for trn_in, trn_out in self.trn_data_generator():
                step_count = 0
                while step_count < self.num_iters:
                    
                    ##update params, output training loss
                    trn_loss_is = self.trn_kernel(trn_out, *trn_in.values())                    
                    if step_count % self.check_every == 0:

                        ##check for improvements
                        val_loss_is = 0
                        for val_in, val_out in self.val_data_generator():
                            val_loss_is += self.loss(val_out, *val_in.values())
                        improved = (val_loss_is < val_loss_was)

                        ##update val loss
                        val_loss_was[improved] = val_loss_is[improved]

                        ##replace old params with better params
                        best_param_values = self.update_best_param_values(best_param_values, improved)

                        ##report on loss history
                        val_history.append(improved.sum())
                        trn_history.append(trn_loss_is)
                        if self.print_stuff:
                            print '====iter: %d' %(step_count)
                            print 'number of improved models: %d' %(val_history[-1])
                            print 'trn error: %0.6f' %(trn_history[-1])
                    
                    step_count += 1

            epoch_count += 1

        ##restore best values of learned params
        set_named_model_params(self.l_model, **{k.name:v for k,v in zip(self.params, best_param_values)})
       

        return self.l_model, val_loss_was, val_history, trn_history

class batch_model_learner_multi( batch_model_learner ):
   
    '''
    batch_model_learner_multi(l_model, model_input_tnsr_dict,trn_data_gen, val_data_gen, num_voxels, num_models)
    
    learn multiple models for a batch of voxels. for each voxel, we have multiple possible models
    to choose from. we use sgd to optimize parameters for each possible model. some methods for selecting
    the best model are added here.
    
    T = number of time points
    G = number of possible models
    V = number of voxels.
    
    lasagne_model is assumed to have an output of dimensions (T, G, V)
    
    loss ~ (G,V)
    
    assumes extra argument "num_models"
    
    
    '''
    
    def __init__(self, *args, **kwargs):
        model_dim_arg_dx = 5 ##we just happen to know this. shit will get fucked up if we get this wrong.
        args = list(args)
        self.num_models = args.pop(model_dim_arg_dx)
        args = tuple(args)
        self.model_dims = kwargs.pop('model_dims', None)
        super(batch_model_learner_multi, self).__init__(*args,**kwargs)
 
        
        self.find_model_dimension()
    
    ##overwrite the training kernel:
    ##construct a gradient update and loss functions to be called iteratively by the learn method
    ##output loss is (G,V)
    def construct_training_kernel( self ):
        voxel_data_tnsr = tnsr.matrix('voxel_data_tnsr')  ##voxel data tensor: (T, V)

        ##get symbolic prediction expression
        pred_expr = lasagne.layers.get_output(self.l_model)  ##voxel prediction tensor: (T,G,V)

        ##generate symbolic loss expression
        trn_diff = voxel_data_tnsr[:,np.newaxis,:]-pred_expr ##difference tensor: shape = (T, V)
        loss_expr = (trn_diff*trn_diff).sum(axis=0)          ##sum squared diffs over time: shape = (G,V)

        ##for *training* error we compute sum of errors along G and V dimensions.
        ##we have to do this because auto-diff requires. a scalar loss function.
        ##BUT: this is fine because gradient w.r.t. one voxel's weights is not affected by loss for any other voxel.
        trn_loss_expr = loss_expr.sum()

        #construct update rule using *training* loss.
        fwrf_update = lasagne.updates.sgd(trn_loss_expr,self.params,learning_rate=self.learning_rate)
        self.trn_kernel = function([voxel_data_tnsr]+self.model_input_tnsr_dict.values(),
                                   trn_loss_expr,
                                   updates=fwrf_update)           
        print 'will update wrt: %s' % (self.params,)
        
        ##compile loss and training functions
        ##NOTE: this is *validation* loss, not summed over voxels, so it has len = num_voxels
        print 'compiling...'
        self.loss = function([voxel_data_tnsr]+self.model_input_tnsr_dict.values(), loss_expr)

    ##get the dimension for the possible models (the "G" dimension) 
    def find_model_dimension( self ):
        if self.model_dims is None:
            self.model_dims = {}
            for p in self.params:
                ##the model dimension should be the one that matches "num_models"
                mdim = [ii for ii,pdim in enumerate(p.shape.eval()) if pdim==self.num_models]
                
                ##if we happen to have multiple dimensions that = "num_models", user must disambiguate
                assert len(mdim)==1, "can't determine model dimension for param %s. supply a 'model_dims' argument" %p.name
                self.model_dims[p.name] = mdim[0]
       
    
    ##overwrite to deal with 2D nature of 'improved_voxels'
    ##if the last gradient step made things better for some voxels, update their parameters
    def update_best_param_values(self, best_param_values, improved_voxels):
        '''
        update_best_param_values(self, best_param_values, improved_voxels)
        improved_voxels ~ (G,V)
        
        '''
        improvement_tuples = np.where(improved_voxels)
        for ii,p in enumerate(self.params):
            vdim = self.voxel_dims[p.name] ##the voxel dimension
            mdim = self.model_dims[p.name]
            s = [slice(None),]*p.ndim      ##create a slicing object with right number of dims
            s[mdim] = improvement_tuples[0]
            s[vdim] = improvement_tuples[1]
            best_param_values[ii][s] = np.copy(p.get_value()[s])   ##keep a record of the best params.
        return best_param_values
    
    def get_min_loss(self, loss):
        min_loss = np.zeros((self.num_voxels))
        argmin_loss = np.zeros((self.num_voxels),dtype='int')
        for ii in range(self.num_voxels):
            min_loss[ii] = np.min(loss[:,ii])
            argmin_loss[ii] = np.argmin(loss[:,ii])
        return min_loss, argmin_loss
    
    
    def select_best_model(self, loss):
        '''
        after learning, submit final val. loss to get a dict. of best model params.
        best param arrays don't have the "model" dimension
        '''
        _,best_model_dx = self.get_min_loss(loss)
        improvement_tuples = [tuple(best_model_dx), tuple(range(self.num_voxels))]
        best_model_params = {}
        for ii,p in enumerate(self.params):
            vdim = self.voxel_dims[p.name] ##the voxel dimension
            mdim = self.model_dims[p.name]
            s = [slice(None),]*p.ndim      ##create a slicing object with right number of dims
            s[mdim] = improvement_tuples[0]
            s[vdim] = improvement_tuples[1]
            best_model_params[p.name] = np.copy(p.get_value()[s])  ##select the best params.
            assert self.num_voxels in best_model_params[p.name].shape, 'voxel dim. not right for best param %s' %p.name
        return best_model_params
 
    