import sys,os
from joblib import Parallel, delayed
import hls4ml
import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from qkeras import QConv2D, QDense, Clip, QActivation
import onnx
import tensorflow_datasets as tfds
from hls4ml.model.profiling import numerical
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas
from plot import getNumpyData
from sklearn.metrics import roc_curve, auc, accuracy_score
from unittest.mock import patch
import time
from optparse import OptionParser

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))
                  
def getQKeras(model,model_name,precision,reuse,intbits_a=6,odir='cnn_projects'):
  
  hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
  hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
  hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

  hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
  hls_config['Model']['ReuseFactor'] = reuse
  
  for Layer in model.layers:
    if isinstance(Layer, tf.keras.layers.Flatten):
      hls_config['LayerName'][Layer.name] = {}  
    hls_config['LayerName'][Layer.name]['ReuseFactor']  = reuse
  hls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'
  print(hls_config)
  
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType']     = 'io_stream'
  cfg['HLSConfig']  = hls_config
  cfg['KerasModel'] = model
  cfg['OutputDir']  = '{}/{}_{}bit_reuse{}'.format(odir,model_name,precision,reuse)
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  
  hls_model = hls4ml.converters.keras_to_hls(cfg)
    
  hls_model.compile()
  
  return hls_model
  
def getBaseline(model,model_name,precision,reuse,intbits_a=6,odir='cnn_projects'):
  # pw = 'ap_fixed<{},{}>'.format(precision,intbits_w)
  po = 'ap_fixed<{},{}>'.format(precision,intbits_a) 
  if precision < 10:
    # pw = 'ap_fixed<{},{}>'.format(precision,4)
    po = 'ap_fixed<{},{}>'.format(precision,4)

  hls_config = {'Model' : {'Precision' : po}}
  hls_config['Model']['ReuseFactor'] = reuse
  hls_config['LayerName']={}
  for Layer in model.layers:
    hls_config['LayerName'][Layer.name] = {}
    hls_config['LayerName'][Layer.name]['ReuseFactor'] = reuse  
  hls_config['LayerName'] = {'output_softmax' : {'Strategy' : 'Stable'}}
  print_dict(hls_config)
  
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType']     = 'io_stream'
  cfg['HLSConfig']  = hls_config
  cfg['KerasModel'] = model
  cfg['OutputDir']  = '{}/{}_{}bit_reuse{}'.format(odir,model_name,precision,reuse)
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  hls_model.compile()
  
  return hls_model
  
def toHLS(p,r,m,doQK=False,intbits_a=0,odir='cnn_projects'):
  if doQK:
    model = tf.keras.models.load_model('models/{}_{}bit_0/model_best.h5'.format(m,p),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
    model = strip_pruning(model)
    hls_model = getQKeras(model=model,model_name=m,precision=p,reuse=r,intbits_a=intbits_a,odir=odir)
  
  else:
    model = tf.keras.models.load_model('models/{}_0/model_best.h5'.format(m),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
    model  = strip_pruning(model)  
    hls_model = getBaseline(model=model,model_name=m,precision=p,reuse=r,intbits_a=intbits_a,odir=odir)
  
  # hls_model.build(csim=False, synth=True, vsynth=True)

if __name__ == '__main__':
  
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename")
  parser.add_option('-m','--model', dest='modelname',default='full')
  parser.add_option('-o','--outdir', dest='outdir',default='cnn_projects')
  parser.add_option('-Q','--doQK',action='store_true', dest='doQKeras'  , default=False, help='do QKeras models')
  parser.add_option('-S','--doScan',action='store_true', dest='doScan'  , default=False, help='do reuse factor scan')

  (options,args) = parser.parse_args()
  
  
  if not os.path.exists('plots'): 
    os.system('mkdir plots')
  if not os.path.exists(options.outdir): 
    os.system('mkdir {}'.format(options.outdir))  

  model_name = options.modelname
  intbits_a = 0
  if not options.doQKeras:
    model = tf.keras.models.load_model('models/{}_0/model_best.h5'.format(model_name),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
    model.summary()
    model  = strip_pruning(model)
    (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=False)
    a = hls4ml.model.profiling.activations_keras(model, x_test[:1000], fmt='summary')
    intbits_a = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], a)))))) + 1)
    print('Starting hls project, using {} int bits'.format(intbits_a))
  
  precision = [16,14,12,10,8,6,4,3,2,1]
  reuse     = [1,2,3,4,5,6]
  precision = [16]
  reuse = [1]
  #
  start = time.time()
  # Parallel(n_jobs=4, backend='multiprocessing')(delayed(toHLS)(i, j,model_name,options.doQKeras,intbits_a,odir=options.outdir) for i in precision for j in reuse)
  end = time.time()
  print('Ended after {:.4f} s'.format(end-start))

  precision = np.flip(precision)
  for p in precision:
   for r in reuse:
     toHLS(p,r,model_name,options.doQKeras,intbits_a,options.outdir)
