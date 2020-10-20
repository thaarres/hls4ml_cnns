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
import cv2
from matplotlib.patches import FancyBboxPatch

def add_logo(ax, fig, zoom, position='upper left', offsetx=10, offsety=10, figunits=False):

  #resize image and save to new file
  img = cv2.imread('logo.jpg', cv2.IMREAD_UNCHANGED)
  im_w = int(img.shape[1] * zoom )
  im_h = int(img.shape[0] * zoom )
  dim = (im_w, im_h)
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  cv2.imwrite( "logo_resized.jpg", resized );

  #read resized image
  im = cv2.imread('logo_resized.jpg')
  im_w = im.shape[1]
  im_h = im.shape[0]

  #get coordinates of corners in data units and compute an offset in pixel units depending on chosen position
  ax_xmin,ax_ymax = 0,0
  offsetX = 0
  offsetY = 0
  if position=='upper left':
   ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[1]
   offsetX,offsetY = offsetx,-im_h-offsety
  elif position=='out left':
   ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[1]
   offsetX,offsetY = offsetx,offsety
  elif position=='upper right':
   ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[1]
   offsetX,offsetY = -im_w-offsetx,-im_h-offsety
  elif position=='out right':
   ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[1]
   offsetX,offsetY = -im_w-offsetx,offsety
  elif position=='bottom left':
   ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[0]
   offsetX,offsetY=offsetx,offsety
  elif position=='bottom right':
   ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[0]
   offsetX,offsetY=-im_w-offsetx,offsety
         
  #transform axis limits in pixel units
  ax_xmin,ax_ymax = ax.transData.transform((ax_xmin,ax_ymax))
  #compute figure x,y of bottom left corner by adding offset to axis limits (pixel units)
  f_xmin,f_ymin = ax_xmin+offsetX,ax_ymax+offsetY
       
  #add image to the plot
  plt.figimage(im,f_xmin,f_ymin)
       
  #compute box x,y of bottom left corner (= figure x,y) in axis/figure units
  if figunits:
   b_xmin,b_ymin = fig.transFigure.inverted().transform((f_xmin,f_ymin))
   #print("figunits",b_xmin,b_ymin)
  else: b_xmin,b_ymin = ax.transAxes.inverted().transform((f_xmin,f_ymin))
  
  #compute figure width/height in axis/figure units
  if figunits: f_xmax,f_ymax = fig.transFigure.inverted().transform((f_xmin+im_w,f_ymin+im_h)) #transform to figure units the figure limits
  else: f_xmax,f_ymax = ax.transAxes.inverted().transform((f_xmin+im_w,f_ymin+im_h)) #transform to axis units the figure limits
  b_w = f_xmax-b_xmin
  b_h = f_ymax-b_ymin

  #set which units will be used for the box
  transformation = ax.transAxes
  if figunits: transformation=fig.transFigure
  
  rectangle = FancyBboxPatch((b_xmin,b_ymin),
                              b_w, b_h,
                  transform=transformation,
                  boxstyle='round,pad=0.004,rounding_size=0.01',
                  facecolor='w',
                  edgecolor='0.8',linewidth=0.8,clip_on=False)
  ax.add_patch(rectangle)
  
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
  ds_train = tfds.load(name=dataset_name, split='train', batch_size=-1)#, data_dir='/afs/cern.ch/user/t/thaarres/tensorflow_datasets/')
  ds_test = tfds.load(name =dataset_name, split='test', batch_size=-1)#, data_dir='/afs/cern.ch/user/t/thaarres/tensorflow_datasets/')

  dataset = tfds.as_numpy(ds_train)
  x_train, y_train = dataset['image'].astype(np.float32), dataset['label']

  dataset = tfds.as_numpy(ds_test)
  x_test, y_test = dataset['image'].astype(np.float32), dataset['label']
  
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
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')
  return (x_train, y_train), (x_test, y_test)

def getAccuracy(name,model,hls_model):
  
  (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=False)
  # wp,ap = numerical(keras_model=model, hls_model=hls_model, X=x_test[:1000])
  # wp.savefig('{}_profile_weights_LayerTypePrecision.pdf'.format(name))
  # ap.savefig('{}_profile_activations_LayerTypePrecision.pdf'.format(name))
  # x_test = x_test[:10]
#   y_test = y_test[:10]
  y_predict        = model    .predict(x_test)
  y_predict_hls4ml = hls_model.predict(x_test)

  accuracy_keras  = float(accuracy_score (y_test, np.argmax(y_predict,axis=1)))
  accuracy_hls4ml = float(accuracy_score(y_test, np.argmax(y_predict_hls4ml,axis=1)))

  print('Accuracy: Keras={} hls4ml={}'.format(accuracy_keras,accuracy_hls4ml))
  hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='plots/plot_model_{}.png'.format(name))
  del x_train, y_train,x_test, y_test
  return accuracy_keras, accuracy_hls4ml
  sys.exit()
  
def getQKeras(baselinemodel,model_name,precision=32):
  model = tf.keras.models.load_model('models/'+model_name+'_{}bit_0/model_best.h5'.format(precision),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
  model.summary()
  model = strip_pruning(model)
  hls_cfg = hls4ml.utils.config_from_keras_model(model)
  hls_cfg['Model']['PackFactor'] = 1 # an integer that divides the image width with no remained
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType'] = 'io_stream'
  cfg['HLSConfig'] = hls_cfg
  cfg['KerasModel'] = model # the model

  cfg['OutputDir'] = 'cnn_projects/'+model_name+'_%ibit'%precision
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  print('Configuration is \n')
  print(cfg)
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  hls_model.compile()
  return model, hls_model
  
def getBaseline(model,model_name,precision=32):
  model = strip_pruning(model)
  # pw = 'ap_fixed<{},{}>'.format(precision,intbits_w)
  po = 'ap_fixed<{},{}>'.format(precision,intbits_a) 
  if precision < 10:
    # pw = 'ap_fixed<{},{}>'.format(precision,4)
    po = 'ap_fixed<{},{}>'.format(precision,4)  
  
  # hls config
  hls_cfg = {'Model' : {'Precision' : po}}
  hls_cfg['LayerName'] = {'output_softmax' : {'Strategy' : 'Stable'}}
  # hls_cfg['LayerType'] = {'Input' : {'Precision' : 'ap_fixed<16,6>'},
  #                         'Dense' : {'Precision' : {'result' : po}},
  #                         'Conv2D': {'Precision' : {'result' : po}},
  #                         'Pooling2D' : {'Precision' : po},
  #                         'BatchNormalization' : {'Precision' : po},
  #                         'Activation' : {'Precision' : {'result' : po}}
  #                        }
  hls_cfg['Model']['PackFactor']  = 1
  hls_cfg['Model']['ReuseFactor'] = 1

  # vivado config
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType']     = 'io_stream'
  cfg['HLSConfig']  = hls_cfg
  cfg['KerasModel'] = model
  cfg['OutputDir']  = 'cnn_projects/baseline_{}_{}'.format(model_name,precision) # wherever you want the project to go
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  hls_model.compile()
  return model, hls_model
        
def getReports(baselinemodel,model_name,p=32):
  
  data = {}
  data['w']= int(p)
  
  if model_name.find('full')!=-1:
    model, hls_model = getBaseline(baselinemodel,model_name,precision=p)
  else:
    model, hls_model = getQKeras(baselinemodel,model_name,precision=p)
    
  data['accuracy_keras'], data['accuracy_hls4ml'] = getAccuracy(model_name,model,hls_model)
  
  indir= '/eos/home-t/thaarres/hls4ml_cnns/synthesized_cnns/{}_{}bit/'.format(model_name,p)
  
  # Get the resources from the logic synthesis report 
  report = open('{}/vivado_synth.rpt'.format(indir))
  lines = np.array(report.readlines())
  data['lut'] = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
  data['ff'] = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
  data['bram'] = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
  data['dsp'] = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
  data['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
  data['ff_rel'] = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
  data['bram_rel'] = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
  data['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])
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
    linestyles = ['solid','dotted','dashed','dashdot']
    lss = itertools.cycle(linestyles)
    plot_lines = []
    fig, ax = plt.subplots()
    plt.grid(color='0.8', linestyle='dotted')
    add_logo(ax, fig, 0.14, position='upper left')
    for i,(data,legend) in enumerate(zip(datas,legends)):
      print(legend)
      print(data)
      ls = next(lss)
      l1, = plt.plot(data['w'], data['dsp_rel'] , color='#a6611a', linestyle=linestyles[i])
      l2, = plt.plot(data['w'], data['lut_rel']       , color='#dfc27d', linestyle=linestyles[i])
      l3, = plt.plot(data['w'], data['ff_rel']        , color ='#80cdc1', linestyle=linestyles[i])
      # l4, = plt.plot(data['w'], data['bram_rel'], color ='#018571', linestyle=linestyles[i])
      plot_lines.append([l1, l2, l3])
    legend1 = plt.legend(plot_lines[0], [r'DSP',r'LUT',r'FF'], loc=1)
    plt.legend([l[0] for l in plot_lines], legends, loc=4)
    plt.gca().add_artist(legend1)
    axes = plt.gca()
    axes.set_ylim([0.01,100.])
    plt.yscale('log')
    plt.xlabel('Bitwidth')
    plt.ylabel('Resource Consumption (fraction/total)')
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
    plt.savefig('plots/scan_resources.pdf')
    plt.close()
   
    lss = itertools.cycle(linestyles)
    plot_lines = []
    plt.clf()
    fig, ax = plt.subplots()
    plt.grid(color='0.8', linestyle='dotted')
    add_logo(ax, fig, 0.14, position='upper left')
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
    axes.set_ylim([1025,1040])
    plt.xlabel('Bitwidth')
    plt.ylabel('Latency (clock cycles)')
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
    plt.savefig('plots/scan_latency_cc.pdf')
    plt.close()

    lss = itertools.cycle(linestyles)
    plot_lines = []
    fig, ax = plt.subplots()
    plt.grid(color='0.8', linestyle='dotted')
    add_logo(ax, fig, 0.14, position='upper left')
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
    axes.set_ylim([0,10])
    plt.xlabel('Bitwidth')
    plt.ylabel('Latency (microseconds)')
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
    plt.savefig('plots/scan_latency_ns.pdf')
    plt.close()


if __name__ == '__main__':
  
  (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=True)
  models = ['full','full_pruned','quant','pruned_quant']
  # models = ['full_final','full_pruned_final']
  legends = ['Full','Pruned','QKeras','QKeras Pruned']
  # legends = ['Full','Pruned']
  datas = []
  
  for model_name in models:
    model = tf.keras.models.load_model('models/{}.h5'.format(model_name),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
    score = model.evaluate(x_test, y_test)
    print('Keras Accuracy {} = {}'.format(model_name,score[1]))
    model = strip_pruning(model)
    intbits_a = 0
    intbits_w = 0
    if model_name.find('full')!=-1:
      a = hls4ml.model.profiling.activations_keras(model, x_test[:100], fmt='summary')
      intbits_a = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], a)))))) + 1)
      # w = hls4ml.model.profiling.weights_keras(model, fmt='summary')
      # intbits_w = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], w)))))) + 1)
    precision = [16,14,12,10,8,6,4,3,2,1]
    precision = [16]
    precision = np.flip(precision)
    data = {'w':[],'accuracy_keras':[],'accuracy_hls4ml':[],  'dsp':[], 'lut':[], 'ff':[],'bram':[],'dsp_rel':[], 'lut_rel':[], 'ff_rel':[],'bram_rel':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
    for p in precision:
      datai = getReports(model,model_name,p)
      for key in datai.keys():
        data[key].append(datai[key])
    data = pandas.DataFrame(data)
    datas.append(data)
  make_plots(datas,legends)
