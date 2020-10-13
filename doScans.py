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

def preprocess(image, label,nclasses=10):
  image = tf.cast(image, tf.float32) / 255.
  label = tf.one_hot(tf.squeeze(label), nclasses)
  return image, label

def toHLS(model,model_name,precision=32):
  
  (img_train, label_train), (img_test, label_test) = tfds.load("svhn_cropped", split=['train', 'test'], batch_size=-1, as_supervised=True,)
  del (img_train, label_train)
  a = hls4ml.model.profiling.activations_keras(model, img_test[:1000], fmt='summary')
  intbits = (np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], a)))))) + 1)
  
  hls_cfg = hls4ml.utils.config_from_keras_model(model, granularity='model')
  hls_cfg['Model']['PackFactor'] = 1 # an integer that divides the image width with no remained
  hls_cfg['Model']['Precision'] = 'ap_fixed<%i,%i>'%(precision,int(intbits))
  print(hls_cfg)
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=hls_cfg, output_dir='model_1/hls4ml_prj')
  # hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
  hls_model.compile()
  score = hls_model.evaluate(test_data)
  print("hlsModel Accuracy {} = {}".format(model_name,score[1]))
  wp,ap = numerical(keras_model=model, hls_model=hls_model, X=img_test[:1000])
  wp.savefig('%s_profile_weights.pdf'%cfg['OutputDir'])
  ap.savefig('%s_profile_activations.pdf'%cfg['OutputDir'])
  
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType'] = 'io_stream'
  cfg['HLSConfig'] = hls_cfg
  cfg['KerasModel'] = model # the model
  cfg['OutputDir'] = model_name.replace(".h5","")+"_bw%i_int%i"%(precision,intbits) # wherever you want the project to go
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  print("Configuration is \n")
  print(cfg)
  
  
  
  indir= '/eos/home-t/thaarres/hls4ml_cnns/synthesized/final_cnn_bw%i/'%precision
  data = {}
  data['w']= int(p)
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
    plt.savefig("scan_resources.pdf")

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
    plt.savefig("scan_latency_cc.pdf")

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
    plt.savefig("scan_latency_ns.pdf")




test_data, test_info = tfds.load("svhn_cropped", split='test', as_supervised=True, with_info=True)
#fig = tfds.show_examples(test_data, test_info)
#fig.savefig("svhn.pdf")
test_data   = test_data .map(preprocess).batch(1024)

models = ["full","full_pruned"]#,"qkeras","qkeras_pruned"]
for model_name in models:
  model = tf.keras.models.load_model("latency_models/{}.h5".format(model_name),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
  score = model.evaluate(test_data)
  print("Accuracy {} = {}".format(model_name,score[1]))
  model = strip_pruning(model)
  print("Starting hls project")
  precision = [16,14,12,10,8,6,4,3,2,1]
# precision = np.flip(precision)
  data = {'accuracy':[], 'w':[], 'dsp':[], 'lut':[], 'ff':[],'bram':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
  for p in precision:
    toHLS(model,model_name,p)
    datai = readReports(model,model_name,p)
#    for key in datai.keys():
#             data[key].append(datai[key])
#
# data = pandas.DataFrame(data)
# make_plots(data)
