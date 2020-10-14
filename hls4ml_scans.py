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
  
def getNumpyData(dataset_name,oneHot=True):
  ds_train = tfds.load(name=dataset_name, split="train", batch_size=-1, data_dir='/afs/cern.ch/user/t/thaarres/tensorflow_datasets/')
  ds_test = tfds.load(name =dataset_name, split="test", batch_size=-1, data_dir='/afs/cern.ch/user/t/thaarres/tensorflow_datasets/')

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
  
  
  (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=False)
  
  
  data = {}
  data['w']= int(p)

  a = hls4ml.model.profiling.activations_keras(model, x_test[:1000], fmt='summary')
  intbits = (np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], a)))))) + 1)
  
  hls_cfg = hls4ml.utils.config_from_keras_model(model, granularity='model')
  hls_cfg['Model']['PackFactor'] = 1 # an integer that divides the image width with no remained
  hls_cfg['Model']['Precision'] = 'ap_fixed<%i,%i>'%(precision,int(intbits))
  print(hls_cfg)
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=hls_cfg, output_dir='model_1/hls4ml_prj')
  # hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
  y_predict = model.predict(x_test[:1000])
  print('Compile and predict')
  hls_model.compile()
  y_predict_hls4ml = hls_model.predict(x_test[:1000])
  print(y_predict_hls4ml)

  data['accuracy_qkeras'] = accuracy_score(y_test[:1000], np.argmax(y_predict,axis=1).reshape(-1,1))
  data['accuracy_hls4ml'] = accuracy_score(y_test[:1000], np.argmax(y_predict_hls4ml,axis=1).reshape(-1,1))
  
      
  print("Accuracy: Keras={} hls4ml={}".format(data['accuracy_qkeras'],data['accuracy_hls4ml']))
  wp,ap = numerical(keras_model=model, hls_model=hls_model, X=x_test[:1000])
  wp.savefig('plots/%s_profile_weights.pdf'%model_name)
  ap.savefig('plots/%s_profile_activations.pdf'%model_name)
  
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType'] = 'io_stream'
  cfg['HLSConfig'] = hls_cfg
  cfg['KerasModel'] = model # the model
  cfg['OutputDir'] = model_name.replace(".h5","")+"_bw%i_int%i"%(precision,int(intbits)) # wherever you want the project to go
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  print("Configuration is \n")
  print(cfg)
  
  
  
  indir= '/eos/home-t/thaarres/hls4ml_cnns/synthesized/{}_bw{}_int{}/'.format(model_name,precision,int(intbits))
  
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

def make_plots(data):
    print(data)
    plt.clf()
    fig,ax = plt.subplots()
    plt.plot(data['w'], data['dsp'] * 10, label=r'DSP (scaled by x10)', color='#a6611a')
    plt.plot(data['w'], data['lut'], label=r'LUT', color='#dfc27d')
    plt.plot(data['w'], data['ff'], label=r'FF', color ='#80cdc1')
    plt.plot(data['w'], data['bram'] * 100, label=r'BRAM (scaled by x100)', color ='#018571')
    plt.legend()
    plt.xlabel('Bitwidth')
    plt.ylabel('Resource Consumption')
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
    plt.savefig("plots/scan_resources.pdf")

    plt.clf()
    fig,ax = plt.subplots()
    plt.plot(data['w'], data['latency_clks'], label=r'Latency', color='#a6611a')
    plt.plot(data['w'], data['latency_ii'], label=r'Initiation Interval', color ='#80cdc1')
    plt.legend()
    plt.figtext(0.125, 0.18,'Post-training quant. (5 ns clock)', wrap=True, horizontalalignment='left',verticalalignment='bottom')
    axes = plt.gca()
    #axes.set_xlim([1.,16])
    axes.set_ylim([6000,8250])
    plt.xlabel('Bitwidth')
    plt.ylabel('Latency (clock cycles)')
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
    plt.savefig("plots/scan_latency_cc.pdf")

    plt.clf()
    fig,ax = plt.subplots()
    plt.plot(data['w'], data['latency_ns'] / 1000., label=r'Latency', color='#a6611a') 
    plt.plot(data['w'], 5 * data['latency_ii'] / 1000., label=r'Initiation Interval', color ='#80cdc1')
    plt.legend()
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
  models = ["full","full_pruned"]#,"qkeras","qkeras_pruned"]
  for model_name in models:
    model = tf.keras.models.load_model("models/{}.h5".format(model_name),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
    score = model.evaluate(x_test, y_test)
    print("Keras Accuracy {} = {}".format(model_name,score[1]))
    model = strip_pruning(model)
    print("Starting hls project")
    precision = [16,14,12,10,8,6,4,3,2,1]
  # precision = np.flip(precision)
    data = {'accuracy':[], 'w':[], 'dsp':[], 'lut':[], 'ff':[],'bram':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
    for p in precision:
      getReports(model,model_name,p)
      # datai = readReports(model,model_name,p)
  #    for key in datai.keys():
  #             data[key].append(datai[key])
  #
  # data = pandas.DataFrame(data)
  # make_plots(data)