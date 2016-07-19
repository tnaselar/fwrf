
# coding: utf-8
import unittest
import string
import lasagne
import numpy as np

from theano import tensor as tnsr
from theano import function, shared
from types import GeneratorType

from fwrf.layers import *
from fwrf.utils import *
from fwrf.learners import *
from fwrf.models import *

# In[ ]:

class TestGetSetShape(unittest.TestCase):
    def setUp(self):
        ##build some random complicated layer
        input_shape = (14, 13, 12, 12)
        num_units = 11
        self.test_layer = lasagne.layers.RecurrentLayer(input_shape,num_units)
    
    def test_param_set_get_shape(self):
        params = lasagne.layers.get_all_params(self.test_layer)
        for p in params:
            val = get_named_params(self.test_layer, p.name)[p.name]
            self.assertEqual(val.get_value().shape, get_named_param_shapes(self.test_layer, p.name)[p.name])
            
    def test_param_names(self):
        pnames = get_model_params_names(self.test_layer)
        self.assertEqual(pnames, [p.name for p in lasagne.layers.get_all_params(self.test_layer)])


class TestRfLayer(unittest.TestCase):
    def setUp(self):
        self.T = 900
        self.D = 90
        self.S = 9
        self.input_shape = (self.T,self.D,self.S,self.S)
        self.V = 10
        self.deg_per_stim = 20
        #this tensor stores feature map
        self.f_map_0 = tnsr.tensor4('f_map_0',dtype='float32')
        ##construct an input layer
        self.input_layer = lasagne.layers.InputLayer(self.input_shape, input_var = self.f_map_0, name='input_layer')
        ##create a real numpy input of all 1's
        self.input_data = np.ones(self.input_shape,dtype='float32')
        
        try:
            self.rf_layer = receptive_field_layer(self.input_layer,
                                                  self.V,
                                                  make_space(self.S),
                                                  self.deg_per_stim, name='rf_layer')
        except:
            print 'could not init rf layer'
    
    ##rf stack should be V,S,S
    def test_rf_layer_make_rf_stack(self):          
        self.assertEqual(self.rf_layer.make_rf_stack().shape, (self.V,self.S,self.S))
    
    ##output should be T,D,V
    def test_rf_layer_output_shape(self):
        output_shape = self.rf_layer.get_output_shape_for(self.input_shape)
        self.assertEqual(output_shape, (self.T,self.D,self.V))
    
    ##grid where rfs are evaluated
    def test_rf_layer_spatial_grid(self):
        self.assertEqual(self.rf_layer.Xm.get_value().shape, (self.S,self.S))
        self.assertEqual(self.rf_layer.Ym.get_value().shape, (self.S,self.S))
    
    ##should all be V
    def test_rf_layer_param_shapes(self):
        self.assertEqual(self.rf_layer.x0.get_value().shape, (self.V,))
        self.assertEqual(self.rf_layer.y0.get_value().shape, (self.V,))
        self.assertEqual(self.rf_layer.sig.get_value().shape, (self.V,))
    
    ##should be T,D,V
    def test_rf_layer_output_function_shape(self):
        rf_layer_expr = lasagne.layers.get_output(self.rf_layer)
        rf_layer_func = function([self.input_layer.input_var], rf_layer_expr)
        self.assertEqual(rf_layer_func(self.input_data).shape,(self.T,self.D,self.V))
        
    ##should go from 0 to 10
    def test_set_value_for_voxel(self):
        self.rf_layer.set_value_for_voxel(voxel_idx=-1,x0=10,y0=10,sig=10)
        self.assertEqual(self.rf_layer.x0.get_value()[-1],10)  ## this is same as next line
        self.assertEqual(get_named_params(self.rf_layer,self.rf_layer.x0.name)[self.rf_layer.x0.name].get_value()[-1],10)
        self.assertEqual(self.rf_layer.y0.get_value()[-1],10)
        self.assertEqual(self.rf_layer.sig.get_value()[-1],10)


class TestCompressiveNonlinearityLayer(unittest.TestCase):
    def setUp(self):
        self.T = 1001
        self.D = 101
        self.V = 11
        self.input_shape = (self.T,self.D,self.V)
        self.test_layer = compressive_nonlinearity_layer(self.input_shape)
    def test_output_shape(self):
        self.assertEqual(self.input_shape, self.test_layer.get_output_shape_for(self.input_shape))


class TestNormalizationLayer(unittest.TestCase):
    def setUp(self):
        self.T = 1001
        self.D = 4
        self.V = 2
        self.x = tnsr.tensor3('x')
        self.input_shape = (self.T,self.D,self.V)
        self.input_layer = lasagne.layers.InputLayer(self.input_shape, input_var=self.x, name='input_layer')
        self.test_input = np.random.random(size=self.input_shape).astype('float32')
        self.test_input[:,0,0] = 1.
        self.mean = np.mean(self.test_input, axis=0)
        self.std = np.std(self.test_input, axis=0)
        self.mask = np.ones((self.D,self.V))
        self.mask[0,0] = 0
        self.test_layer = normalization_layer(self.input_layer, mean=self.mean, stdev = self.std, mask = self.mask)
        
    def test_output_shape(self):
        self.assertEqual(self.input_shape, self.test_layer.get_output_shape_for(self.input_shape))
    
    def test_out_values(self):
        output_func = function([self.x], [lasagne.layers.get_output(self.test_layer)])
        y = output_func(self.test_input)[0]
        ##shape is right
        self.assertEqual(self.input_shape, y.shape)
        ##degenerate entry set to 0
        self.assertTrue(np.all(y[:,0,0]==0))
        ##mean is 0
        np.testing.assert_almost_equal(np.mean(y,axis=0), 0, decimal=5)
        ##stdev is 1 for all but degnerate entry
        np.testing.assert_almost_equal(np.std(y,axis=0).ravel()[1:], 1, decimal=6)


class TestFeatureWeightsLayer(unittest.TestCase):
    def setUp(self):
        T,D,V = 1001,4,20
        self.input_shape = (T,D,V)
        self.x = tnsr.tensor3('x')
        self.input_layer = lasagne.layers.InputLayer(self.input_shape, input_var=self.x, name='input_layer')
        self.test_input = np.random.random(size=self.input_shape).astype('float32')
        self.test_layer = feature_weights_layer(self.input_layer)
        self.layer_func = function([self.x], lasagne.layers.get_output(self.test_layer))
        self.y = self.layer_func(self.test_input)
    
    def test_output(self):
        self.assertEqual((self.input_shape[0], self.input_shape[-1]), self.test_layer.get_output_shape_for(self.input_shape))
        self.assertEqual((self.input_shape[0], self.input_shape[-1]), self.y.shape)
        
    def test_set_weight_for_voxel(self):
        self.assertIsNone(self.test_layer.set_weight_for_voxel(NU = None))
        voxel_idx = [0,1,-2,-1]
        NU = [1,1] ##this raise value error because of mismatch
        with self.assertRaises(ValueError):
            self.test_layer.set_weight_for_voxel(voxel_idx = voxel_idx, NU = NU)     
        NU = np.random.random((self.input_shape[1], len(voxel_idx))).astype('float32')
        self.test_layer.set_weight_for_voxel(voxel_idx = voxel_idx, NU=NU)
        np.testing.assert_array_equal(self.test_layer.NU.get_value()[:,voxel_idx], NU)


class TestPredictionMenuLayer( unittest.TestCase ):
    def setUp(self):
        T,D,G,V = 100, 11, 5, 3
        x = tnsr.tensor3('x')
        test_input = np.random.random((T,D,G)).astype('float32')
        test_layer = lasagne.layers.InputLayer((T,D,G), input_var=x)
        test_layer = prediction_menu_layer(test_layer,V)
        test_layer_expr = lasagne.layers.get_output(test_layer) 
        test_layer_func = function([x], test_layer_expr)
        
        ##record
        self.T,self.D,self.G,self.V = T,D,G,V
        self.test_layer = test_layer
        self.test_input = test_input
        self.test_layer_func = test_layer_func
    
    def test_output_shape(self):
        self.assertEqual((self.T,self.G,self.V), self.test_layer.get_output_shape_for(self.test_input.shape))
        self.assertEqual((self.T,self.G,self.V), self.test_layer_func(self.test_input).shape)


class TestBatchModelLearner(unittest.TestCase):
    
    def setUp(self):
        Ttrn,Tval,D,S,V = 5000,200,5,2,3
        self.V = V
        self.Ttrn = Ttrn
        self.Tval = Tval
        model_input_tnsr_dict = {}
        model_input_tnsr_dict['fmap0'] = tnsr.tensor4('fmap0')
        input_layer = lasagne.layers.InputLayer((None,D,S,S), input_var = model_input_tnsr_dict['fmap0'])
        
        self.true_W = np.random.random((D*S*S,V)).astype('float32')
        true_model = lasagne.layers.DenseLayer(input_layer,num_units = V, W=self.true_W)
        out_func = function([model_input_tnsr_dict['fmap0']], lasagne.layers.get_output(true_model))
        
        ##create training/val data using true model
        trn_in = {}
        trn_in['fmap0'] = np.random.random((Ttrn,D,S,S)).astype('float32')
        trn_out = out_func(trn_in['fmap0'])
        trn_data_gen = lambda: (yield trn_in, trn_out)
        val_in = {}
        val_in['fmap0'] = np.random.random((Tval,D,S,S)).astype('float32')
        val_out = out_func(val_in['fmap0'])
        val_data_gen = lambda: (yield val_in,val_out)
        
        ##test basic dimensions, parameter assignment
        self.learn_nothing = batch_model_learner(true_model, model_input_tnsr_dict,trn_data_gen, val_data_gen, self.V)
        
        ##test learning of "b": start at b=1, see if it will learn b = 0
        test_b_model = lasagne.layers.DenseLayer(input_layer,
                                                 num_units = V,
                                                 W=self.true_W,
                                                 b=np.ones((self.V,)).astype('float32'))
        self.learn_b = batch_model_learner(test_b_model,
                                           model_input_tnsr_dict,
                                           trn_data_gen,
                                           val_data_gen,
                                           self.V,
                                           learn_these_params=['b'],
                                           check_every=100,
                                           learning_rate= 0.00001,
                                           num_iters = 400,
                                           print_stuff=False)

        ##test learning of "W": start at W = some other random values
        test_W_model = lasagne.layers.DenseLayer(input_layer,num_units = V, W=np.random.random((D*S*S,V)).astype('float32'))
        self.learn_W = batch_model_learner(test_b_model,
                                           model_input_tnsr_dict,
                                           trn_data_gen,
                                           val_data_gen,
                                           self.V,
                                           learn_these_params=['W'],
                                           check_every=100,
                                           learning_rate= 10e-15,
                                           num_iters = 90000,
                                           print_stuff=False)
        
        ##test learning of "W" and "b"
        test_both_model = lasagne.layers.DenseLayer(input_layer,
                                                    num_units = V,
                                                    W=np.random.random((D*S*S,V)).astype('float32'),
                                                    b=np.ones((self.V,)).astype('float32'))
        self.learn_both = batch_model_learner(test_both_model,
                                              model_input_tnsr_dict,
                                              trn_data_gen,
                                              val_data_gen,
                                              self.V,
                                              check_every=100,
                                              learning_rate= 0.00001,
                                              num_iters = 15000,
                                              print_stuff=False)
    
    def test_num_voxels(self):
        self.assertEqual(self.learn_nothing.num_voxels, self.V)
    
    def test_voxel_dims(self):
        self.assertDictEqual(self.learn_nothing.voxel_dims, {'W': 1, 'b':0})        
        
    def test_loss_func(self):
        trn_in,trn_out = self.learn_nothing.trn_data_generator().next()
        trn_loss = self.learn_nothing.loss(trn_out,*trn_in.values())
        self.assertEqual(trn_out.shape, (self.Ttrn, self.V))
        self.assertEqual(trn_loss.shape, (self.V,))
        np.testing.assert_array_equal(trn_loss, np.zeros((self.learn_nothing.num_voxels,)))
    
    def test_learn_b(self):
        new_model,val_loss,val_hist,trn_hist = self.learn_b.learn()
        np.testing.assert_array_almost_equal(new_model.b.get_value(), np.zeros((self.V)), decimal=5)
        
    def test_learn_W(self):
        new_model,val_loss,val_hist,trn_hist = self.learn_W.learn()
        np.testing.assert_array_almost_equal(new_model.W.get_value(), self.true_W, decimal=5)
    
    def test_learn_both(self):
        new_model,val_loss,val_hist,trn_hist = self.learn_both.learn()
        np.testing.assert_array_almost_equal(new_model.b.get_value(), np.zeros((self.V)), decimal=4)
        np.testing.assert_array_almost_equal(new_model.W.get_value(), self.true_W, decimal=4)
        

class TestBatchModelLearnerMulti(unittest.TestCase):
    
    def setUp(self):
        Ttrn,Tval,D,G,V = 5000,200,5,2,3
        model_input_tnsr_dict = {}
        model_input_tnsr_dict['fmap0'] = tnsr.tensor3('fmap0')
        input_layer = lasagne.layers.InputLayer((None,D,G), input_var = model_input_tnsr_dict['fmap0'])
        
        true_NU = np.random.random((D,G,V)).astype('float32')
        true_rf = np.random.randint(0,high=G,size=V)
        best_model_matrix = [slice(None), tuple(true_rf), tuple(range(V))]
        true_model = prediction_menu_layer(input_layer,V, NU=true_NU)
        
        
        tmp_out_func = function([model_input_tnsr_dict['fmap0']], lasagne.layers.get_output(true_model))
        out_func = lambda x: tmp_out_func(x)[best_model_matrix]


        trn_in = {}
        trn_in['fmap0'] = np.random.random((Ttrn,D,G)).astype('float32')
        trn_out = out_func(trn_in['fmap0'])
        trn_data_gen = lambda: (yield trn_in, trn_out)
        val_in = {}
        val_in['fmap0'] = np.random.random((Tval,D,G)).astype('float32')
        val_out = out_func(val_in['fmap0'])
        val_data_gen = lambda: (yield val_in,val_out)   
        
        ##test basic dimensions, parameter assignment
        learn_nothing = batch_model_learner_multi(true_model,
                                                  model_input_tnsr_dict,
                                                  trn_data_gen,
                                                  val_data_gen,
                                                  V,
                                                  G,
                                                  model_dims=None)
        val_loss = learn_nothing.loss(val_out,*val_in.values())        

        
        ##test learning of NU
        test_model = prediction_menu_layer(input_layer,V, NU=np.random.random((D,G,V)).astype('float32'))
        learn_something = batch_model_learner_multi(test_model,
                                                    model_input_tnsr_dict,
                                                    trn_data_gen,
                                                    val_data_gen,
                                                    V,
                                                    G,
                                                    check_every=100,
                                                    learning_rate= 0.0000001,
                                                    num_iters = 150000,
                                                    print_stuff=False)
        
 
        ##record everything needed in self
        self.V = V
        self.Ttrn = Ttrn
        self.Tval = Tval
        self.G = G
        self.input_layer = input_layer
        self.true_NU = true_NU
        self.true_rf = true_rf
        self.learn_nothing = learn_nothing
        self.learn_something = learn_something
        self.val_in = val_in
        self.val_out = val_out
 
    
    def test_num_voxels(self):
        self.assertEqual(self.learn_nothing.num_voxels, self.V)
    
    def test_voxel_dims(self):
        self.assertDictEqual(self.learn_nothing.voxel_dims, {'feature_weights': 2})   
        
    def test_data_generator(self):
        val_in, val_out = self.learn_nothing.val_data_generator().next()
        self.assertEqual(val_out.shape, (self.Tval, self.V))
        
    def test_loss_func_and_best_model_selection(self):
        
        val_loss = self.learn_nothing.loss(self.val_out,*self.val_in.values())        
        self.assertEqual(val_loss.shape, (self.G, self.V,))
        
        min_val_loss,argmin_val_loss = self.learn_nothing.get_min_loss(val_loss)

        
        np.testing.assert_array_equal(min_val_loss, np.zeros(self.learn_nothing.num_voxels))
        np.testing.assert_array_equal(argmin_val_loss, self.true_rf)
        
        best_model_params = self.learn_nothing.select_best_model(val_loss)
        for vv in range(self.V):
            np.testing.assert_array_equal(best_model_params['feature_weights'][:,vv], self.true_NU[:,self.true_rf[vv],vv])
            
    def test_learn_something(self):
        new_model,val_loss,val_hist,trn_hist = self.learn_something.learn()
        best_model_params = self.learn_something.select_best_model(val_loss)
        for vv in range(self.V):
            np.testing.assert_array_almost_equal(best_model_params['feature_weights'][:,vv],
                                                 self.true_NU[:,self.true_rf[vv],vv],
                                                 decimal=2)
    

class TestRfModelSpace( unittest.TestCase ):
    
    def setUp(self):
        T,D,V = 20,13,3
        self.V = V
        self.nmaps = 10
        self.D = D
        self.T = T
        self.deg_per_stim = 10
        self.feature_map_dict = {}
        ##create 10 feature maps where resolution happens to be name
        for ii in range(1,self.nmaps+1):
            input_name = 'fmap_%0.2d' % (ii)
            self.feature_map_dict[input_name] = np.random.random((T,D,ii,ii)).astype('float32')
        
        self.rfms = rf_model_space(self.feature_map_dict, self.deg_per_stim, self.V)
        
        self.cal_data_generator = lambda: (yield self.feature_map_dict)
    
    def test_basic_attributes(self):
        self.assertEqual(self.rfms.D, self.D*self.nmaps)
    
    def test_construct_model_space_tensor(self):
        mst = self.rfms.construct_model_space_tensor(*self.feature_map_dict.values())
        self.assertEqual(mst.shape, (self.T, self.rfms.D, self.V))
        
    def test_normalize(self):
        self.rfms.normalize(self.cal_data_generator)
        mst = self.rfms.construct_model_space_tensor(*self.feature_map_dict.values())
        np.testing.assert_array_almost_equal(np.mean(mst,axis=0), np.zeros((self.rfms.D,self.V)), decimal=6)
        np.testing.assert_array_almost_equal(np.std(mst,axis=0), np.ones((self.rfms.D,self.V)), decimal=6)

class TestFwrf( unittest.TestCase ):
    
    def setUp(self):
        
        ##params
        Ttrn,Tval,D,V,nmaps = 2003,301,17,4,8
        deg_per_stim = 20
        sf = 10

        ##feature maps
        trn_feature_map_dict = {}
        for ii in range(1,nmaps+1):
            input_name = 'fmap_%0.2d' % (ii)
            trn_feature_map_dict[input_name] = np.random.random((Ttrn,D,sf*ii,sf*ii)).astype('float32')

        val_feature_map_dict = {}
        for ii in range(1,nmaps+1):
            input_name = 'fmap_%0.2d' % (ii)
            val_feature_map_dict[input_name] = np.random.random((Tval,D,sf*ii,sf*ii)).astype('float32')


        ##true rfs
        bound = int((deg_per_stim-3)/2.)
        true_rf_params = {k:np.random.randint(-bound,high=bound, size=V).astype('float32') for k in ['x0','y0']}
        true_rf_params['sig'] = np.random.randint(1,high=bound,size=V).astype('float32')
        ##true feature weights
        true_NU = np.random.random((D*nmaps,V)).astype('float32')

        ##fwrf model
        true_model = fwrf(val_feature_map_dict,deg_per_stim,V,rf_init=true_rf_params, NU=true_NU)
        true_model.normalize(lambda: (yield trn_feature_map_dict))

        ##true outputs, trn/val
        trn_voxel_activity = true_model.predict(trn_feature_map_dict).astype('float32')
        val_voxel_activity = true_model.predict(val_feature_map_dict).astype('float32')

        ##data generator: note these are functions that *return* generators, so we can reboot the generator whenever.
        chunk_size = 100
        trn_data_gen = lambda: (({k:v[ii:ii+chunk_size,:,:,:] for k,v in trn_feature_map_dict.iteritems()}, trn_voxel_activity[ii:ii+chunk_size,:]) for ii in range(0,Ttrn,chunk_size))
        val_data_gen = lambda: (({k:v[ii:ii+chunk_size,:,:,:] for k,v in val_feature_map_dict.iteritems()}, val_voxel_activity[ii:ii+chunk_size,:]) for ii in range(0,Tval,chunk_size))       

        ##rf grid for coarse training
        deg_per_radius = (1,deg_per_stim,12)
        spacing = 2
        rf_grid_df = make_rf_table(deg_per_stim,deg_per_radius,spacing,pix_per_stim = None)
        G = rf_grid_df.shape[0]
        rf_grid = {}
        rf_grid['x0'] = rf_grid_df.x_deg.values.astype('float32')
        rf_grid['y0'] = rf_grid_df.y_deg.values.astype('float32')
        rf_grid['sig'] = rf_grid_df.deg_per_radius.values.astype('float32')

        ##record what is needed
        self.true_model = true_model
        self.trn_data_gen = trn_data_gen
        self.val_data_gen = val_data_gen
        self.true_NU = true_NU
        self.true_rf_params = true_rf_params
        self.rf_grid = rf_grid
        self.Tval = Tval
        self.V = V
        self.Ttrn = Ttrn
        self.chunk_size = chunk_size
        self.D = D
        self.nmaps = nmaps
        self.G = G


    ##test fwrf output dimensions
    def test_fwrf_output_dimensions(self):
        inp,outp = self.val_data_gen().next()
        self.assertEqual(outp.shape,(self.chunk_size, self.V))
        inp,outp = self.trn_data_gen().next()
        self.assertEqual(outp.shape,(self.chunk_size, self.V))
        self.assertEqual(self.D*self.nmaps, self.true_model.D)

    ##loss is 0 for fwrf with true params
    def test_fwrf_true_output(self):
        inp,outp = self.val_data_gen().next()
        learner = batch_model_learner(self.true_model.fwrf,
                                      self.true_model.input_var_dict,
                                      self.trn_data_gen,
                                      self.val_data_gen,
                                      self.V)
        val_loss = learner.loss(outp, *inp.values())
        np.testing.assert_array_almost_equal(val_loss, np.zeros(val_loss.shape), decimal = 7)
    
    ##test create proxy and test proxy shape
    def test_create_proxy_net(self):
        pxnet,_ = self.true_model._build_proxy_network((None,self.D*self.nmaps,self.G))
        out_shape = pxnet.get_output_shape_for((self.Ttrn,self.true_model.D,self.G))
        self.assertEqual(out_shape, (self.Ttrn,self.G,self.V))
        
    ##test construct model space tensor
    def test_construct_model_space_tensor( self ):
        _,mst_gen = self.true_model._build_rf_model_space_tensor(self.trn_data_gen,
                                                                 self.val_data_gen,
                                                                 self.rf_grid,
                                                                 consolidate=False)
        self.assertEqual(mst_gen().next()[0]['mst'].shape, (self.chunk_size,self.true_model.D,self.G))
        self.assertEqual(mst_gen().next()[1].shape, (self.chunk_size, self.V))
        
        _,mst_gen = self.true_model._build_rf_model_space_tensor(self.trn_data_gen,
                                                                 self.val_data_gen,
                                                                 self.rf_grid,
                                                                 consolidate=True)
        self.assertEqual(mst_gen().next()[0]['mst'].shape, (self.Tval,self.true_model.D,self.G))
        self.assertEqual(mst_gen().next()[1].shape, (self.Tval, self.V))
        
    
    ##test coarse learning using rf grid
    def test_fwrf_train_me( self ):
        ##coarse
        _=self.true_model.train_me(self.trn_data_gen, self.val_data_gen,
                              coarse=True,
                              rf_grid=self.rf_grid,
                              learning_rate=10e-8,
                              epochs=10,
                              num_iters = 1,
                              check_every=1,
                              print_stuff=True,
                              check_dims=False,
                              normalize=True,consolidate=True)
        ##fine
        _=self.true_model.train_me(self.trn_data_gen, self.val_data_gen,
                              fine=True,
                              learning_rate=10e-8,
                              epochs=7,
                              num_iters = 1,
                              check_every=1,
                              print_stuff=True,
                              check_dims=False)          
        
        np.testing.assert_array_almost_equal(self.true_model.rf_layer.sig.get_value().astype('float32'),
                                             self.true_rf_params['sig'],
                                             decimal=2)
        np.testing.assert_array_almost_equal(self.true_model.rf_layer.x0.get_value().astype('float32'),
                                             self.true_rf_params['x0'],
                                             decimal=2)
        np.testing.assert_array_almost_equal(self.true_model.rf_layer.y0.get_value().astype('float32'),
                                             self.true_rf_params['y0'],
                                             decimal=2)
    
if __name__ == '__main__':
    unittest.main()

