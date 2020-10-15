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
def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

def toHLS(precision=32):
  m=model_stripped
  #hls_cfg = hls4ml.utils.config_from_keras_model(m,granularity='type')
  #hls_cfg['Model'] = {}
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
  cfg['KerasModel'] = m # the model
   #hls_cfg['Model']['Precision'] = 'ap_fixed<{},{}>'.format(precision,intbits_w)
   #hls_cfg['LayerType']['Activation']['Precision'] = 'ap_fixed<{},{}>'.format(precision,intbits_a)
   #hls_cfg['Model']['Precision']=pw 
   #hls_cfg['LayerType']['Input'] = {'Precision':'ap_fixed<16,6>'}
   #hls_cfg['LayerType']['Conv2D']['Precision']['result'] = po
   #hls_cfg['LayerType']['Dense']['Precision']['result']  = po
   #hls_cfg['LayerType']['MaxPooling2D']['Precision']  = po
   #hls_cfg['LayerType']['BatchNormalization']['Precision']['result']  = po
   #print(po)
   #hls_cfg['Model']['Precision']['weight'] = pw
   #hls_cfg['Model']['Precision']['bias']   = pw
   #hls_cfg['Model']['Precision']['result'] = po
  cfg['OutputDir'] = model_name.replace(".h5","")+"_bw%i"%(precision) # wherever you want the project to go
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  print("Configuration is \n")
  print_dict(hls_cfg)
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  hls_model.build(csim=False,synth=True, vsynth=True)
  #hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='plot_model_{}.png'.format(precision))
  #wp,ap = numerical(keras_model=m, hls_model=hls_model, X=img_test[:5000])
  #add_logo(ax, fig, 0.14, position='upper right')
  #wp.savefig('%s_profile_weights.pdf'%cfg['OutputDir'])
  #ap.savefig('%s_profile_activations.pdf'%cfg['OutputDir'])
  #hls_model.build(csim=False, synth=True, vsynth=True) 

def readReports(indir,p):
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
    #plt.figtext(0.125, 0.18,'Post-training quant. (5 ns clock)', wrap=True, horizontalalignment='left',verticalalignment='bottom')
    #axes = plt.gca()
    #axes.set_xlim([1.,16])
    #axes.set_ylim([0,200000])
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

if __name__ == '__main__':
    doParallel = int(sys.argv[2])
    model_name = str(sys.argv[1])
    model = tf.keras.models.load_model("/data/thaarres/hls4ml_docker/hls4ml_cnns/latency_models/"+model_name,custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
    model.summary()
    model_stripped = strip_pruning(model)
    (img_train, label_train), (img_test, label_test) = getNumpyData('svhn_cropped',oneHot=False)
    a = hls4ml.model.profiling.activations_keras(model_stripped, img_test[:1000], fmt='summary')
    intbits_a = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], a)))))) + 1)
    w = hls4ml.model.profiling.weights_keras(model_stripped, fmt='summary')
    intbits_w = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], w)))))) + 1)
    print("Starting hls project, using {} int bits for weights+bias and {} int bits for outputs".format(intbits_a,intbits_w))
    precision = [16,14,12,10,8,6,4,3,2,1]
    data = {'w':[], 'dsp':[], 'lut':[], 'ff':[], 'bram':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
    #if doParallel:
    Parallel(n_jobs=10, backend='multiprocessing')(delayed(toHLS)(i) for i in precision)
    #precision = np.flip(precision)
    #else:
     #   for p in precision:
      #      toHLS(p)
    #for p in precision:
   #     datai = readReports(model_name.replace(".h5","")+"_bw%i"%(p),p)
  #      for key in datai.keys():
 #           data[key].append(datai[key])
#
    #data = pandas.DataFrame(data)
    #make_plots(data)
   # data.to_csv(r'data_%s.csv'%model_name, index = False)
