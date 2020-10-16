import sys
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
from hls4ml_scans import getNumpyData
from sklearn.metrics import roc_curve, auc, accuracy_score

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

def toHLS(model,precision=32):
  (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=False)
  
  #get model precision
  pw = 'ap_fixed<{},{}>'.format(precision,intbits_w)
  po = 'ap_fixed<{},{}>'.format(precision,intbits_a) 
  if precision < 10:
    pw = 'ap_fixed<{},{}>'.format(precision,4)
    po = 'ap_fixed<{},{}>'.format(precision,4)  
  
  # hls config
  hls_cfg = {'Model' : {'Precision' : pw}}
  hls_cfg['LayerName'] = {'output_softmax' : {'Strategy' : 'Stable'}}
  hls_cfg['LayerType'] = {'Input' : {'Precision' : 'ap_fixed<16,6>'},
                          'Dense' : {'Precision' : {'result' : po}},
                          'Conv2D' : {'Precision' : {'result' : po}},
                          'Pooling2D' : {'Precision' : po},
                          'BatchNormalization' : {'Precision' : po},
                          'Activation' : {'Precision' : {'result' : po}}
                         }
  hls_cfg['Model']['PackFactor']  = 1
  hls_cfg['Model']['ReuseFactor'] = 1
  
  # vivado config
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType']     = 'io_stream'
  cfg['HLSConfig']  = hls_cfg
  cfg['KerasModel'] = model
  cfg['OutputDir']  = model_name.replace(".h5","")+"_bw%i"%(precision) # wherever you want the project to go
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  print_dict(hls_cfg)
  
  #hls model
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  for layer in hls_cfg['LayerName'].keys():
      hls_cfg['LayerName'][layer]['Trace'] = True
  hls_model.compile()
  hls4ml_pred, hls4ml_trace = hls_model.trace(x_test[:1000])
  keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, x_test[:1000])
  y_hls = hls_model.predict(x_test[:1000])
  
  for layer in hls_cfg['LayerName'].keys():
    print("Keras layer {}, first sample:".format(layer))
    print(keras_trace[layer][0])
    print("hls4ml layer {}, first sample:".format(layer))
    print(hls4ml_trace[layer][0])
  
  print('Compile and predict')
  x_test = x_test[:100]
  y_test = y_test[:100]
  y_predict        = model    .predict(x_test)
  y_predict_hls4ml = hls_model.predict(x_test)
  print("y_predict = ", y_predict[2])
  print("y_predict_hls4ml = ", y_predict_hls4ml[2])
  print("y_test = ", y_test[2])
  print("arg ypred = ", np.argmax(y_predict[2]))
  print("arg ypredhls = ", np.argmax(y_predict_hls4ml[2]))
  data['accuracy_keras'] = accuracy_score (y_test, np.argmax(y_predict,axis=1))
  data['accuracy_hls4ml'] = accuracy_score(y_test, np.argmax(y_predict_hls4ml,axis=1))

  print("Accuracy: Keras={} hls4ml={}".format(data['accuracy_keras'],data['accuracy_hls4ml']))
  hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='plot_model_{}.png'.format(precision))

  hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='plot_model_{}.png'.format(precision))
  wp,ap = numerical(keras_model=model, hls_model=hls_model, X=x_test[:1000])

  wp.savefig('%s_profile_weights_LayerTypePrecision.pdf'%cfg['OutputDir'])
  ap.savefig('%s_profile_activations_LayerTypePrecision.pdf'%cfg['OutputDir'])
  #hls_model.build(csim=False, synth=True, vsynth=True) 

if __name__ == '__main__':
    model_name = str(sys.argv[1])
    model = tf.keras.models.load_model("models/"+model_name,custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
    model.summary()
    model  = strip_pruning(model)
    (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=False)
    a = hls4ml.model.profiling.activations_keras(model, x_test[:1000], fmt='summary')
    intbits_a = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], a)))))) + 1)
    w = hls4ml.model.profiling.weights_keras(model, fmt='summary')
    intbits_w = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], w)))))) + 1)
    print("Starting hls project, using {} int bits for weights+bias and {} int bits for outputs".format(intbits_a,intbits_w))
    precision = [16,14,12,10,8,6,4,3,2,1]
    precision = [16]
    data = {'w':[], 'dsp':[], 'lut':[], 'ff':[], 'bram':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
    #Parallel(n_jobs=10, backend='multiprocessing')(delayed(toHLS)(i) for i in precision)
    #precision = np.flip(precision)
    for p in precision:
      toHLS(model,p)
    #for p in precision:
   #     datai = readReports(model_name.replace(".h5","")+"_bw%i"%(p),p)
  #      for key in datai.keys():
 #           data[key].append(datai[key])
#
    #data = pandas.DataFrame(data)
    #make_plots(data)
   # data.to_csv(r'data_%s.csv'%model_name, index = False)
