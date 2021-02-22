import sys,os
from joblib import Parallel, delayed
import hls4ml
import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from qkeras import QConv2D, QDense, Clip, QActivation, QBatchNormalization
import onnx
import tensorflow_datasets as tfds
from hls4ml.model.profiling import numerical
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas
from plot import getNumpyData, add_logo
from sklearn.metrics import roc_curve, auc, accuracy_score
from unittest.mock import patch
import time
from optparse import OptionParser
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import mplhep as hep
plt.style.use(hep.style.CMS)
import matplotlib as mpl 
mpl.rcParams["yaxis.labellocation"] = 'center'
mpl.rcParams["xaxis.labellocation"] = 'center'

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
  # hls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'
  hls_config['LayerName']['output']['Strategy'] = 'Stable'
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
    model = tf.keras.models.load_model('models/{}/model_best.h5'.format(m),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation,'QBatchNormalization':QBatchNormalization})
    model = strip_pruning(model)
    hls_model = getQKeras(model=model,model_name=m,precision=p,reuse=r,intbits_a=intbits_a,odir=odir)
  
  else:
    model = tf.keras.models.load_model('models/{}/model_best.h5'.format(m),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
    model  = strip_pruning(model)  
    hls_model = getBaseline(model=model,model_name=m,precision=p,reuse=r,intbits_a=intbits_a,odir=odir)
  (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=True)
  wp,ap = numerical(model=model, hls_model=hls_model, X=x_test[:1000])
  ap.axes[0].set_title("")
  add_logo(ap.axes[0], ap, 0.3, position='upper left')
  labels = [item.get_text().replace('batch_normalization','Batch norm.').replace('max_pooling2d','Max Pooling').replace('_',' ').capitalize() for item in ap.axes[0].get_yticklabels()]
  ap.axes[0].set_yticklabels(labels)
  ap.axes[0].set_xlabel('Output')
  ap.axes[0].set_xlim([10.455191523E-13,64])
  ap.savefig('plots/Profile_{}_activations.pdf'.format(model.name))
  del ap
  wp.axes[0].set_title("")
  labels = [item.get_text().replace('batch_normalization','Batch norm.').replace('max_pooling2d','Max Pooling').replace('_',' ').replace('0 0','0, w').replace('0 1','0, b').replace('1 0','1, w').replace('1 1','1, b').replace('2 0','2, w').replace('2 1','2, b').replace('3 0','3, w').replace('3 1','3, b').replace('4 0','4, w').replace('4 1','4, b').replace('5 0','5, w').replace('5 1','5, b').replace('output dense 0','output, w').replace('output dense 1','output, b').replace('norm. 0','norm., w').replace('norm. 1','norm., b').replace(', b,',' 1,').replace('output 0','output, w').replace('output 1','output, b').capitalize() for item in wp.axes[0].get_yticklabels()]
  # labels = [item.replace('Dense','Fused dense + b n.').replace('Output Fused dense + b n.','Output dense') for item in labels]
  wp.axes[0].set_yticklabels(labels)
  wp.axes[0].set_xlim([0.0000009,64])
  wp.axes[0].set_xlabel('Weight')
  add_logo(wp.axes[0], wp, 0.3, position='upper left')
  wp.savefig('plots/Profile_{}_weights.pdf'.format(model.name))
  # hls_model.build(csim=False, synth=True, vsynth=True)

if __name__ == '__main__':
  
  parser = OptionParser()
  parser.add_option('-m','--model', dest='modelname',default='full')
  parser.add_option('-Q','--doQK',action='store_true', dest='doQKeras'  , default=False, help='do QKeras models')
  parser.add_option('-S','--doScan',action='store_true', dest='doScan'  , default=False, help='do reuse factor scan')

  (options,args) = parser.parse_args()
  

  reuse     = [1,2,3,4,5,6]
  reuse     = [1]
  start = time.time()
  # Parallel(n_jobs=4, backend='multiprocessing')(delayed(toHLS)(0, j,options.modelname,options.doQKeras,intbits_a=0,odir='cnn_projects') for j in reuse)
  
 
  for r in reuse:
    toHLS(16, r,options.modelname,options.doQKeras,intbits_a=6,odir='cnn_projects')
  end = time.time()
  print('Ended after {:.4f} s'.format(end-start))
 
  
