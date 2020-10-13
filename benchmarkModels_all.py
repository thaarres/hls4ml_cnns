import sys
import hls4ml
import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from qkeras import QConv2D, QDense, Clip, QActivation
import onnx
import tensorflow_datasets as tfds
from hls4ml.model.profiling import numerical

from os import listdir
from os.path import isfile, join

def toHLS(m,precision=32):
  hls_cfg = hls4ml.utils.config_from_keras_model(m)
  hls_cfg['Model']['PackFactor'] = 1 # an integer that divides the image width with no remained
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType'] = 'io_stream'
  cfg['HLSConfig'] = hls_cfg
  cfg['KerasModel'] = m # the model
  cfg['OutputDir'] = model_name.replace(".h5","")+"_bw%i"%precision # wherever you want the project to go
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  print(cfg)
  hls_model = hls4ml.converters.keras_to_hls(cfg)

  #(img_train, label_train), (img_test, label_test) = tfds.load("svhn_cropped", split=['train', 'test'], batch_size=-1, as_supervised=True,)
  #del (img_train, label_train)

  #wp,ap = numerical(keras_model=m, hls_model=hls_model, X=img_test[:1000])
  #wp.savefig('%s_profile_weights.pdf'%model_name)
  #ap.savefig('%s_profile_activations.pdf'%model_name)
  hls_model.build(csim=False, synth=True, vsynth=True) 
  
indir_name = str(sys.argv[1])
path = "/data/thaarres/hls4ml_docker/hls4ml_cnns/"+indir_name
print("Starting hls project")
files = [f for f in listdir(path) if isfile(join(path, f))]
for f in files:
 model_name = f
 model = tf.keras.models.load_model(path+f,custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
 model.summary()
 model_stripped = strip_pruning(model)
 toHLS(model_stripped)
