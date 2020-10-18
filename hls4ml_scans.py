#import setGPU
import sys
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
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc, accuracy_score
import itertools

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))
              
def getNumpyData(dataset_name,oneHot=True):
  ds_train = tfds.load(name=dataset_name, split="train", batch_size=-1)#, data_dir='/afs/cern.ch/user/t/thaarres/tensorflow_datasets/')
  ds_test = tfds.load(name =dataset_name, split="test", batch_size=-1)#, data_dir='/afs/cern.ch/user/t/thaarres/tensorflow_datasets/')

  dataset = tfds.as_numpy(ds_train)
  x_train, y_train = dataset["image"].astype(np.float32), dataset["label"]

  dataset = tfds.as_numpy(ds_test)
  x_test, y_test = dataset["image"].astype(np.float32), dataset["label"]
  
  if len(x_train.shape) == 3:
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

  x_train /= 255.0
  x_test /= 255.0

  #Get mean and std over all images and subtract/divide
  x_mean = np.mean(x_train, axis=0)
  x_std  = np.std(x_train, axis = 0)
  x_train = (x_train-x_mean)/x_std
  x_test  = (x_test-x_mean)/x_std

  
  if oneHot:
    nb_classes = np.max(y_train) + 1
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
  #
  print(x_train.shape[0], "train samples")
  print(x_test.shape[0], "test samples")
  return (x_train, y_train), (x_test, y_test)
    
def getReports(model,model_name,precision=32):
  outname = model_name.replace('.h5','')+'_bw{}'.format(precision)
  (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=False)
  
  data = {}
  data['w']= int(p)

  pw = 'ap_fixed<{},{}>'.format(precision,intbits_w)
  po = 'ap_fixed<{},{}>'.format(precision,intbits_a) 
  if precision < 10:
    pw = 'ap_fixed<{},{}>'.format(precision,4)
    po = 'ap_fixed<{},{}>'.format(precision,4)  
  
  hls_cfg = {'Model' : {'Precision' : pw}}
  hls_cfg['LayerType'] = {'Input' : {'Precision' : 'ap_fixed<16,6>'},
                        'Dense' : {'Precision' : {'result' : po}},
                        'Conv2D' : {'Precision' : {'result' : po}},
                        'MaxPooling2D' : {'Precision' : {'result' : po}},
                        'BatchNormalization' : {'Precision' : {'result' : po}},
                        'Activation' : {'Precision' : {'result' : po}}}
  hls_cfg['Model']['PackFactor'] = 1 # an integer that divides the image width with no remained
  hls_cfg['Model']['ReuseFactor'] = 1
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType'] = 'io_stream'
  cfg['HLSConfig'] = hls_cfg
  cfg['KerasModel'] = model # the model
  cfg['OutputDir'] = outname
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  print("Configuration is \n")
  print_dict(hls_cfg)
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  hls_model.compile()
  
  print('Compile and predict')
  x_test = x_test[:1000]
  y_test = y_test[:1000]
  y_predict        = model    .predict(x_test)
  y_predict_hls4ml = hls_model.predict(x_test)
  print("y_predict = ", y_predict[2])
  print("y_predict_hls4ml = ", y_predict_hls4ml[2])
  print("y_test = ", y_test[2])
  print("arg ypred = ", np.argmax(y_predict[2]))
  print("arg ypredhls = ", np.argmax(y_predict_hls4ml[2]))
  data['accuracy_keras'] = accuracy_score (y_test, np.argmax(y_predict,axis=1).reshape(-1,1))
  data['accuracy_hls4ml'] = accuracy_score(y_test, np.argmax(y_predict_hls4ml,axis=1).reshape(-1,1))
  print(y_test- np.argmax(y_predict_hls4ml,axis=1))
      
  print("Accuracy: Keras={} hls4ml={}".format(data['accuracy_keras'],data['accuracy_hls4ml']))
  wp,ap = numerical(keras_model=model, hls_model=hls_model, X=x_test[:1000])
  wp.savefig('plots/{}_profile_weights.pdf'.format(outname))
  ap.savefig('plots/{}_profile_activations.pdf'.format(outname))
  
  indir= '/eos/home-t/thaarres/hls4ml_cnns/models_synt/{}_bw{}/'.format(model_name,precision)
  
  # Get the resources from the logic synthesis report 
  report = open('{}/vivado_synth.rpt'.format(indir))
  lines = np.array(report.readlines())
  data['lut'] = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
  data['ff'] = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
  data['bram'] = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
  data['dsp'] = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
  report.close()
  
  # Get the latency from the Vivado HLS report
  report = open('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
  lines = np.array(report.readlines())
  lat_line = lines[np.argwhere(np.array(['Latency (clock cycles)' in line for line in lines])).flatten()[0] + 6]
  data['latency_clks'] = int(lat_line.split('|')[2])
  data['latency_ns'] = float(lat_line.split('|')[2])*5.0
  data['latency_ii'] = int(lat_line.split('|')[4])
  return data

def make_plots(datas,legends):
    linestyles = ["solid","dotted"]
    lss = itertools.cycle(linestyles)
    plot_lines = []
    plt.clf()
    fig,ax = plt.subplots()
    for i,(data,legend) in enumerate(zip(datas,legends)):
      print(data)
      ls = next(lss)
      l1, = plt.plot(data['w'], data['dsp'] * 10  , color='#a6611a', linestyle=linestyles[i])
      l2, = plt.plot(data['w'], data['lut']       , color='#dfc27d', linestyle=linestyles[i])
      l3, = plt.plot(data['w'], data['ff']        , color ='#80cdc1', linestyle=linestyles[i])
      l4, = plt.plot(data['w'], data['bram'] * 100, color ='#018571', linestyle=linestyles[i])
      plot_lines.append([l1, l2, l3, l4])
    legend1 = plt.legend(plot_lines[0], [r'DSP (scaled by x10)',r'LUT',r'FF',r'BRAM (scaled by x100)'], loc=1)
    plt.legend([l[0] for l in plot_lines], legends, loc=4)
    plt.gca().add_artist(legend1)
    
    plt.xlabel('Bitwidth')
    plt.ylabel('Resource Consumption')
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
    plt.savefig("plots/scan_resources.pdf")
   
    lss = itertools.cycle(linestyles)
    plot_lines = []
    plt.clf()
    for i,(data,legend) in enumerate(zip(datas,legends)):
      print(data)
      
      ls = next(lss)
      l1, = plt.plot(data['w'], data['latency_clks'], color='#a6611a', linestyle=linestyles[i])
      l2, = plt.plot(data['w'], data['latency_ii'] , color ='#80cdc1', linestyle=linestyles[i])
      plot_lines.append([l1, l2])
    legend1 = plt.legend(plot_lines[0], [r'Latency',r'Initiation Interval'], loc=1)
    plt.legend([l[0] for l in plot_lines], legends, loc=4)
    plt.gca().add_artist(legend1)
    plt.figtext(0.125, 0.18,'Post-training quant. (5 ns clock)', wrap=True, horizontalalignment='left',verticalalignment='bottom')
    axes = plt.gca()
    #axes.set_xlim([1.,16])
    axes.set_ylim([6000,8250])
    plt.xlabel('Bitwidth')
    plt.ylabel('Latency (clock cycles)')
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
    plt.savefig("plots/scan_latency_cc.pdf")

    lss = itertools.cycle(linestyles)
    plot_lines = []
    plt.clf()
    for i,(data,legend) in enumerate(zip(datas,legends)):
      print(data)
      ls = next(lss)
      l1, = plt.plot(data['w'], data['latency_ns'] / 1000., color='#a6611a', linestyle=linestyles[i])
      l2, = plt.plot(data['w'], 5 * data['latency_ii'] / 1000., color ='#80cdc1', linestyle=linestyles[i])
      plot_lines.append([l1, l2])
    legend1 = plt.legend(plot_lines[0], [r'Latency',r'Initiation Interval'], loc=1)
    plt.legend([l[0] for l in plot_lines], legends, loc=4)
    plt.gca().add_artist(legend1)  
    plt.figtext(0.125, 0.18,'Post-training quant. (5 ns clock)', wrap=True, horizontalalignment='left',verticalalignment='bottom')
    axes = plt.gca()
    #axes.set_xlim([1.,16])
    axes.set_ylim([30,45])
    plt.xlabel('Bitwidth')
    plt.ylabel('Latency (microseconds)')
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
    plt.savefig("plots/scan_latency_ns.pdf")



if __name__ == "__main__":
  
  
  (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=True)
  models = ["full_final","full_pruned_final"]#,"qkeras","qkeras_pruned"]
  legends = ['Full','Pruned']
  datas = []
  for model_name in models:
    model = tf.keras.models.load_model("models/{}.h5".format(model_name),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
    score = model.evaluate(x_test, y_test)
    print("Keras Accuracy {} = {}".format(model_name,score[1]))
    model = strip_pruning(model)
    a = hls4ml.model.profiling.activations_keras(model, x_test[:1000], fmt='summary')
    intbits_a = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], a)))))) + 1)
    w = hls4ml.model.profiling.weights_keras(model, fmt='summary')
    intbits_w = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], w)))))) + 1)
    print("Starting hls project")
    precision = [16,14,12,10,8,2]
    precision = [16]
    precision = np.flip(precision)
    data = {'w':[],'accuracy_keras':[],'accuracy_hls4ml':[],  'dsp':[], 'lut':[], 'ff':[],'bram':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
    for p in precision:
      datai = getReports(model,model_name,p)
      for key in datai.keys():
        data[key].append(datai[key])
    data = pandas.DataFrame(data)
    datas.append(data)
  make_plots(datas,legends)
