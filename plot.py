#import setGPU
import sys,os
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
from matplotlib.pyplot import imread 
from matplotlib.patches import FancyBboxPatch
from optparse import OptionParser
from pathlib import Path
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.ticker as mtick
import mplhep as hep

plt.style.use(hep.style.CMS)

import matplotlib as mpl 
mpl.rcParams["yaxis.labellocation"] = 'center'
mpl.rcParams["xaxis.labellocation"] = 'center'
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['lines.markeredgewidth'] = 2.0 
# mpl.rcParams['lines.markeredgewidth'] = 0.
mpl.rcParams['xtick.minor.top'] = False    # draw x axis top minor ticks
mpl.rcParams['xtick.minor.bottom'] = False    # draw x axis bottom minor ticks
mpl.rcParams['ytick.minor.left'] = True    # draw x axis top minor ticks
mpl.rcParams['ytick.minor.right'] = True    # draw x axis bottom minor ticks

colz = ['#003f5c','#7a5195','#ef5675','#ffa600','#67a9cf','#016c59']

mstyle = ['o','^','s','v']*3
fstyle = ['#003f5c','None','#ef5675','None','#67a9cf','None']
fstyle = colz
def cc2mus(x):
    return x * 5. / 1000.

def rmus2cc(x):
    return x * 1000. / 5.
    
def add_logo2(ax,fig,zoom=0.10):
  ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[1]
  logo = mpimg.imread('logo.jpg')

  imagebox = OffsetImage(logo, zoom=zoom)
  
  # fig.figimage(imagebox, 0, fig.bbox.ymax)
  ab = AnnotationBbox(imagebox, (ax_xmin*0.87,ax_ymax*0.92),frameon=False)#,xycoords='axes pixels')
  ax.add_artist(ab)
  
def add_logo(ax, fig, zoom, position='upper left', offsetx=10, offsety=10, figunits=False):

  #resize image and save to new file
  img = cv2.imread('logo.jpg', cv2.IMREAD_UNCHANGED)
  im_w = int(img.shape[1] * zoom )
  im_h = int(img.shape[0] * zoom )
  dim = (im_w, im_h)
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  cv2.imwrite( "logo_resized.jpg", resized );

  #read resized image
  im = imread('logo_resized.jpg')
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
  fig.figimage(im,f_xmin,f_ymin)
       
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
                  edgecolor='w',linewidth=0.8,clip_on=False)
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

def getAccuracy(model,hls_model,x,y):
  # wp,ap = numerical(model=model, hls_model=hls_model, X=x[:1000])
  # wp.savefig('{}_profile_weights_LayerTypePrecision.pdf'.format(model.name))
  # ap.savefig('{}_profile_activations_LayerTypePrecision.pdf'.format(model.name))
  # sys.exit()
  # y_predict        = m    .predict(x_test)
  # y_predict_hls4ml = hm.predict(x_test)
  #hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='plots/plot_model_{}.png'.format(model.name))
  y_predict        = model.predict(x[:10000])
  y_predict_hls4ml = hls_model.predict(x[:10000])
  y = y[:10000]
  accuracy_keras  = float(accuracy_score (y, np.argmax(y_predict,axis=1)))
  accuracy_hls4ml = float(accuracy_score (y, np.argmax(y_predict_hls4ml,axis=1)))

  return accuracy_keras, accuracy_hls4ml
  
def getQKeras(model,model_name,precision,x_test, y_test):
  for layer in model.layers:
          if hasattr(layer, "kernel_quantizer"):
              print(layer.name, "kernel:", str(layer.kernel_quantizer_internal), "bias:", str(layer.bias_quantizer_internal))
          elif hasattr(layer, "quantizer"):
              print(layer.name, "quantizer:", str(layer.quantizer))
  hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
  hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
  hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

  hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  print(model_name)
  if model_name.find('AutoQ')!=-1:
    hls_config['LayerName']['output']['Strategy'] = 'Stable'
  else:
    hls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType']     = 'io_stream'
  cfg['HLSConfig']  = hls_config
  cfg['KerasModel'] = model
  cfg['OutputDir']  = 'cnn_projects/{}_bit{}'.format(model_name,precision)
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  hls_model.compile()
  acc_keras, acc_hls4ml = getAccuracy(model,hls_model,x_test, np.argmax(y_test,axis=1))
  return acc_keras, acc_hls4ml
  
def getBaseline(model,model_name,precision,x_test, y_test):
  pw = 'ap_fixed<{},{}>'.format(precision,intbits_w)
  po = 'ap_fixed<{},{}>'.format(precision,intbits_a) 
  if precision < 10:
    pw = 'ap_fixed<{},{}>'.format(precision,4)
    po = 'ap_fixed<{},{}>'.format(precision,4)
  
  # hls config
  # config = {}
  # config['Model'] = {}
  # config['Model']['Precision']= {}
  # config['Model']['Precision']['default'] = 'ap_fixed<16,6>'
  # config['Model']['Precision']['result'] = po
  # config['Model']['Precision']['weight'] = pw
  
  hls_cfg = {'Model' : {'Precision' : po}}
  hls_cfg['LayerName'] = {'output_softmax' : {'Strategy' : 'Stable'}}
  # config['LayerType'] = {'Input' : {'Precision' : 'ap_fixed<16,6>'},
  #                         'Dense' : {'Precision' : {'result' : po}},
  #                         'Conv2D': {'Precision' : {'result' : po}},
  #                         'Pooling2D' : {'Precision' : po},
  #                         'BatchNormalization' : {'Precision' : po},
  #                         'Activation' : {'Precision' : {'result' : po}}
  #                        }
  hls_cfg['Model']['PackFactor']  = 1
  hls_cfg['Model']['ReuseFactor'] = 1
  print_dict(hls_cfg)

  # hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir='test')
  cfg = hls4ml.converters.create_vivado_config()
  cfg['IOType']     = 'io_stream'
  cfg['HLSConfig']  = hls_cfg
  cfg['KerasModel'] = model
  cfg['OutputDir']  = 'cnn_projects/{}_bit{}'.format(model_name,precision)
  cfg['XilinxPart'] = 'xcvu9p-flgb2104-2l-e'
  hls_model = hls4ml.converters.keras_to_hls(cfg)
  hls_model.compile()
  acc_keras, acc_hls4ml = getAccuracy(model,hls_model,x_test, np.argmax(y_test,axis=1))
  return acc_keras, acc_hls4ml
  
  hls_model.compile()
  acc_keras, acc_hls4ml = getAccuracy(model,hls_model,x_test, np.argmax(y_test,axis=1))
  return acc_keras, acc_hls4ml
        
def getReports(mname,p,x_test, y_test):
  
  data_ = {}
  
  indir = 'projects/{}_{}bit_reuse1/'.format(mname,p)
  report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
  report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
  
  if report_vsynth.is_file() and report_csynth.is_file():
    print('Found valid vsynth and synth! Fetching numbers')
    
  
    if mname.find('full')!=-1:
      data_['w']= int(p)
      if int(p)>9:
        model_  = tf.keras.models.load_model('models/{}.h5'.format(mname),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
        model_ = strip_pruning(model_)
        data_['accuracy_keras'], data_['accuracy_hls4ml'] = getBaseline(model_,mname,p,x_test, y_test)
      else:
        data_['accuracy_keras'] = 0.01
        data_['accuracy_hls4ml'] = 0.01
          
    else:
      data_['w']= int(p+1)
      print('models/'+mname+'_{}bit_0/model_best.h5'.format(p))
      model_ = tf.keras.models.load_model('models/'+mname+'_{}bit_0/model_best.h5'.format(p),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
      model_ = strip_pruning(model_)
      data_['accuracy_keras'], data_['accuracy_hls4ml'] = getQKeras(model_,mname,p,x_test, y_test)
    
    
    print('Accuracy: Keras={} hls4ml={}'.format(data_['accuracy_keras'], data_['accuracy_hls4ml']))
    
    
    # Get the resources from the logic synthesis report 
    with report_vsynth.open() as report:
      lines = np.array(report.readlines())
      data_['lut']     = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
      data_['ff']      = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
      data_['bram']    = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
      data_['dsp']     = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
      data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
      data_['ff_rel']  = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
      data_['bram_rel']= float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
      data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])
    
    with report_csynth.open() as report:
      lines = np.array(report.readlines())
      lat_line = lines[np.argwhere(np.array(['Latency (clock cycles)' in line for line in lines])).flatten()[0] + 6]
      data_['latency_clks'] = int(lat_line.split('|')[2])
      data_['latency_ns']   = float(lat_line.split('|')[2])*5.0
      data_['latency_ii']   = int(lat_line.split('|')[4])
    
    return data_
  
  else:
    # print('No synth reports found! Returning empty data frame')
    return data_
    
def make_plots(datas,legends,outname='per_model',doAcc=True,datas_AQ=None,legs_AQ=None):
    linestyles = ['solid','dotted','dashed','dotted']
    
    # Plotting resources
    resources = ['dsp','lut','ff','bram']
    
    for resource in resources:
      fig = plt.figure()#figsize=(6,4), constrained_layout=False) 
      
      if datas_AQ is not None:
        gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[5,1])
        ax = fig.add_subplot(gs[0,0])
      else:
        ax = fig.add_subplot()  
      for i,(data,legend) in enumerate(zip(datas,legends)):                                       
        l1, = plt.plot(data['w'], data['{}_rel'.format(resource)] , label = legend , color = colz[i], marker=mstyle[i], markerfacecolor=fstyle[i], markeredgecolor=colz[i])
      if datas_AQ is not None:
        aqlab = ['AQ', 'AQP']
        for j,data_AQ in enumerate(datas_AQ):
          plt.plot(-1, data_AQ['{}_rel'.format(resource)], color = colz[-(j+1)], marker='D', label = aqlab[j],  linestyle = 'None')
      plt.legend(loc='upper left',frameon=False)
      plt.gca().set_prop_cycle(None)
      axes = plt.gca()
      axes.set_xlim([1.,17.])
      if resource.find('dsp')!=-1:
        axes.set_ylim([0.,130.])
      if resource.find('lut')!=-1:
        axes.set_ylim([0.,100.])
      if resource.find('bram')!=-1:
        axes.set_ylim([0.,5.])
      if resource.find('ff')!=-1:
        axes.set_ylim([0.,5.])
        if outname.find('reuse')!=-1:
          axes.set_ylim([0.,7.])
      # if resource.find('lut')!=-1:
      #   axes.set_ylim([0.,5.])
      # plt.yscale('log')
      plt.xlabel('Bit width')
      plt.ylabel('{} consumption [%]'.format(resource.upper()))
      # plt.figtext(0.125, 0.518,'Xilinx VU9P', wrap=True, horizontalalignment='left',verticalalignment='bottom')
      plt.locator_params(axis='y', nbins=4)
      plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
      # plt.gca().get_yaxis().set_major_locator(MaxNLocator(integer=True))
      # add_logo(ax, fig, 0.14, position='upper right')
      if datas_AQ is not None:
        ax1 = fig.add_subplot(gs[0,1], sharey=ax)
        gs.update(wspace=0.0, hspace=0.00)
        plt.gca().set_prop_cycle(None)
        x = 0
        for i,data_AQ in enumerate(datas_AQ):
          plt.plot(x, data_AQ['{}_rel'.format(resource)], color = colz[-(i+1)], marker='D',  linestyle = 'None')
          x+=1
        ax1.set_xlim((-0.5, len(datas_AQ) - 0.5))
        ax1.get_yaxis().set_visible(False)
        plt.xticks(list(range(len(legs_AQ))), legs_AQ, fontsize=20)
      add_logo(ax, fig, 0.25, position='upper right', offsetx=10, offsety=10, figunits=False)
      plt.savefig('plots/{}_scan_{}.pdf'.format(outname,resource))
      plt.close()
    
    # Plotting latency
    resources = ['latency_clks','latency_ii']
    for resource in resources:
      fig = plt.figure()#figsize=(6,4), constrained_layout=False) 
      if datas_AQ is not None:
        gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[5,1])
        ax = fig.add_subplot(gs[0,0])
      else:
        ax = fig.add_subplot()
      for i,(data,legend) in enumerate(zip(datas,legends)):
        l1, = plt.plot(data['w'], data[resource], color = colz[i], linestyle=linestyles[0], label = legend, marker=mstyle[i], markerfacecolor=fstyle[i], markeredgecolor=colz[i])
      if datas_AQ is not None:
        aqlab = ['AQ', 'AQP']
        for j,data_AQ in enumerate(datas_AQ):
          plt.plot(-1, data_AQ['{}'.format(resource)], color = colz[-(j+1)], marker='D', label = aqlab[j],  linestyle = 'None')

      plt.legend(loc='upper left',frameon=False)
      plt.gca().set_prop_cycle(None)
      # plt.gca().add_artist(legend1)
      # plt.figtext(0.125, 0.518,'5 ns clock, Xilinx VU9P', wrap=True, horizontalalignment='left',verticalalignment='bottom')
      axes = plt.gca()
      axes.set_xlim([1.,17.])
      if resource == 'latency_clks':
        ylab = 'Latency (clock cycles)'
      else:
        ylab = 'Initiation Interval (clock cycles)'
      #axes.set_xlim([1.,16])
      if outname.find('reuse')!=-1:
        axes.set_ylim([800,14000])
      else:
        if resource == 'latency_clks':
          axes.set_ylim([1020,1065])
        else:
          axes.set_ylim([1027,1032])
      ax.set_ylabel(ylab)
      if datas_AQ is None:
        secax = axes.secondary_yaxis('right', functions=(cc2mus, rmus2cc))
        secax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
        secax.yaxis.set_major_locator(plt.MaxNLocator(3))
        secax.set_ylabel(ylab.replace('clock cycles','$\mu$s'))
      ax.set_xlabel('Bit width')
      plt.locator_params(axis='y', nbins=4)
      plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
      plt.gca().get_yaxis().set_major_locator(MaxNLocator(integer=True))
      ax.yaxis.set_major_locator(plt.MaxNLocator(6))
      # add_logo(ax, fig, 0.14, position='upper right')
      if datas_AQ is not None:
        ax1 = fig.add_subplot(gs[0,1], sharey=ax)
        gs.update(wspace=0.0, hspace=0.00)
        plt.gca().set_prop_cycle(None)
        x = 0
        for i,data_AQ in enumerate(datas_AQ):
          plt.plot(x, data_AQ[resource], color = colz[-(i+1)], marker='D',  linestyle = 'None')
          x += 1
        ax1.set_xlim((-0.5, len(datas_AQ) - 0.5))
        ax1.get_yaxis().set_visible(False)
        secax = ax1.secondary_yaxis('right', functions=(cc2mus, rmus2cc))
        secax.set_ylabel(ylab.replace('clock cycles','$\mu$s'))
        secax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        secax.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.xticks(list(range(len(legs_AQ))), legs_AQ, fontsize=20)
      add_logo(ax, fig, 0.25, position='upper right')
      
      plt.savefig('plots/{}_{}.pdf'.format(outname,resource))
      plt.close()
      
    # for resource in resources:
    #   fig = plt.figure(figsize=(6,4), constrained_layout=False)
    #   if datas_AQ is not None:
    #     gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[5,1])
    #     ax = fig.add_subplot(gs[0,0])
    #   else:
    #     ax = fig.add_subplot()
    #
    #   for i,(data,legend) in enumerate(zip(datas,legends)):
    #     l1, = plt.plot(data['w'], 0.005 * data[resource], color = colz[i], linestyle=linestyles[0], label=legend, marker='8')
    #     # l2, = plt.plot(data['w'], 5 * data['latency_ii'] / 1000., color = colz[i], linestyle=linestyles[1], marker='8')
    #     # plot_lines.append([l1, l2])
    #   # legend1 = plt.legend(plot_lines[0], [r'Latency',r'Initiation Interval'], loc='center left',frameon=False)
    #   plt.legend( loc='upper left',frameon=False)
    #   plt.gca().set_prop_cycle(None)
    #   # plt.gca().add_artist(legend1)
    #   # plt.figtext(0.125, 0.518,'5 ns clock, Xilinx VU9P', wrap=True, horizontalalignment='left',verticalalignment='bottom')
    #   axes = plt.gca()
    #   #axes.set_xlim([1.,16])
    #   if outname.find('reuse')!=-1:
    #     axes.set_ylim([0.0,60.])
    #   else:
    #     axes.set_ylim([5.1,5.2])
    #
    #   plt.xlabel('Bit width')
    #   if resource == 'latency_clks':
    #     plt.ylabel('Latency ($\mu$s)')
    #     axes.set_ylim([5.0,5.4])
    #     if outname.find('reuse')!=-1:
    #       axes.set_ylim([0.0,60.])
    #   else:
    #     plt.ylabel('Initiation Interval ($\mu$s)')
    #     axes.set_ylim([5.1,5.2])
    #     if outname.find('reuse')!=-1:
    #       axes.set_ylim([0.0,60.])
    #   plt.locator_params(axis='y', nbins=4)
    #   plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
    #   # plt.gca().get_yaxis().set_major_locator(MaxNLocator(integer=True))
    #   # add_logo(ax, fig, 0.14, position='upper right')
    #   if datas_AQ is not None:
    #     ax1 = fig.add_subplot(gs[0,1], sharey=ax)
    #     plt.gca().set_prop_cycle(None)
    #     gs.update(wspace=0.0, hspace=0.00)
    #     x = 0
    #     for i,data_AQ in enumerate(datas_AQ):
    #       plt.plot(x, 0.005 * data_AQ[resource], color = colz[-(i+1)], linestyle=linestyles[0], marker='8')
    #       x += 1
    #     ax1.set_xlim((-0.5, len(datas_AQ) - 0.5))
    #     ax1.get_yaxis().set_visible(False)
    #     plt.xticks(list(range(len(legs_AQ))), legs_AQ, fontsize=10)
    #   add_logo(ax, fig, 0.14, position='upper right')
    #   plt.savefig('plots/{}_{}_scan_latency_ns.pdf'.format(outname,resource))
    #   plt.close()
    
    #Plotting accuracy
    if doAcc:
      plot_lines = []
      fig = plt.figure()#figsize=(6,4), constrained_layout=False) 
      if datas_AQ is not None:
        gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[5,1])
        ax = fig.add_subplot(gs[0,0])
      else:
        ax = fig.add_subplot()
      
      for i,(data,legend) in enumerate(zip(datas,legends)):
        if legend.find('B')==-1:
           l1, = plt.plot(data['w'], data['accuracy_keras'] , color =colz[i], linestyle=linestyles[1], marker=mstyle[i], markerfacecolor='None', markeredgecolor=colz[i])
        else:
          l1 = None
        l2, = plt.plot(data['w'], data['accuracy_hls4ml'], color =colz[i], linestyle=linestyles[0], marker=mstyle[i], markerfacecolor=fstyle[i], markeredgecolor=colz[i])
        plot_lines.append([l1, l2])
      if datas_AQ is not None:
        aqlab = ['AQ', 'AQP']
        for j,data_AQ in enumerate(datas_AQ):
          l3, = plt.plot(-1, data_AQ['{}'.format(resource)], color = colz[-(j+1)], marker='D', label = aqlab[j],  linestyle = 'None')  
          plot_lines.append([l3, l3])
        legends = legends+aqlab
      legend1 = plt.legend(plot_lines[2], ['Keras','hls4ml'], loc='lower right',frameon=False)
      plt.legend([l[1] for l in plot_lines], legends, loc='upper left',frameon=False)
      plt.gca().set_prop_cycle(None)
      plt.gca().add_artist(legend1)
      axes = plt.gca()
      axes.set_xlim([1.,17.])
      #axes.set_xlim([1.,16])
      axes.set_ylim([0.5,1.2])

      plt.xlabel('Bit width')
      plt.ylabel('Accuracy')
      # plt.figtext(0.6, 0.218,'Xilinx VU9P', wrap=True, horizontalalignment='left',verticalalignment='bottom')
      plt.locator_params(axis='y', nbins=4)
      plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
      # plt.gca().get_yaxis().set_major_locator(MaxNLocator(integer=True))
      # add_logo(ax, fig, 0.14, position='upper right')
      if datas_AQ is not None:
        ax1 = fig.add_subplot(gs[0,1], sharey=ax)
        plt.gca().set_prop_cycle(None)
        gs.update(wspace=0.0, hspace=0.00)
        x = 0
        for i,data_AQ in enumerate(datas_AQ):
          plt.plot(x, data_AQ['accuracy_keras'], color = colz[-(i+1)] , marker ='D',  linestyle = 'None',fillstyle='none')
          plt.plot(x, data_AQ['accuracy_hls4ml'], color = colz[-(i+1)], marker='D',  linestyle = 'None')
          x += 1
        ax1.set_xlim((-0.5, len(datas_AQ) - 0.5))
        ax1.get_yaxis().set_visible(False)
        plt.xticks(list(range(len(legs_AQ))), legs_AQ, fontsize=20)
      add_logo(ax, fig, 0.25, position='upper right')
      plt.savefig('plots/{}_scan_accuracy.pdf'.format(outname))
      plt.close()

def make_2Dplots(datas,legends,outname='per_model'):
    linestyles = ['solid','dotted','dashed','dotted']
    
    
    # plt.grid(color='0.8', linestyle='dotted')
    
    resources = ['dsp','lut','ff','bram']
    for resource in resources:
      fig, ax = plt.subplots()
      add_logo(ax, fig, 0.3, position='upper right')
      for j,(data,legend) in enumerate(zip(datas,legends)): 
                                              
        l1, = plt.plot(data['r'], data['{}_rel'.format(resource)] , label = legend , color = colz[j], linestyle=linestyles[0], markersize=10, marker=mstyle[j])
      plt.legend(loc='upper left',frameon=False)
      # plt.gca().add_artist(legend1)
      axes = plt.gca()

      plt.xlabel('R')
      plt.ylabel('{} consumption [%]'.format(resource.upper()))
      if resource.find('dsp')!=-1:
        axes.set_ylim([0.,140.])
      if resource.find('ff')!=-1:
        axes.set_ylim([0.,8.])
      if resource.find('lut')!=-1:
        axes.set_ylim([0.,110.])  
      if resource.find('bram')!=-1:
        axes.set_ylim([2.,4.])  
      # plt.figtext(0.125, 0.518,'Xilinx VU9P', wrap=True, horizontalalignment='left',verticalalignment='bottom')
      plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
      plt.gca().get_yaxis().set_major_locator(MaxNLocator(integer=True))
      # add_logo(ax, fig, 0.14, position='upper right')
      plt.savefig('plots/{}_reuse_vs_{}.pdf'.format(outname,resource))
      plt.close()
      
    resource= 'latency_clks'
    fig, ax = plt.subplots()
    add_logo(ax, fig, 0.3, position='upper right')
    for i,(data,legend) in enumerate(zip(datas,legends)):                                       
      l1, = plt.plot(data['r'], data[resource] , label = legend , color = colz[i], linestyle=linestyles[0], marker=mstyle[i], markersize=10)
    plt.legend(loc='upper left',frameon=False)
    axes = plt.gca()

    plt.xlabel('R')
    plt.ylabel('Latency (clock cycles)')
    axes.set_ylim([1000.,7500.]) 
    # plt.figtext(0.125, 0.518,'Xilinx VU9P', wrap=True, horizontalalignment='left',verticalalignment='bottom')
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
    # add_logo(ax, fig, 0.14, position='upper right')
    plt.savefig('plots/{}_reuse_vs_{}.pdf'.format(outname,resource))
    plt.close()
               
def getSynthData(model_name,p,r):
  data_ = {}
  
  indir = 'projects/{}/'.format(model_name)
  report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
  report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
  
  if report_vsynth.is_file() and report_csynth.is_file():
    # print('Found valid vsynth and synth for {}! Fetching numbers'.format(model_name))
    data_['w']= int(p)
    if model_name.find('quant')!=-1:
      data_['w']= int(p+1)
    data_['r']= int(r)
  
    # Get the resources from the logic synthesis report 
    with report_vsynth.open() as report:
      lines = np.array(report.readlines())
      data_['lut']     = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
      data_['ff']      = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
      data_['bram']    = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
      data_['dsp']     = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
      data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
      data_['ff_rel']  = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
      data_['bram_rel']= float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
      data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])
    
    with report_csynth.open() as report:
      lines = np.array(report.readlines())
      lat_line = lines[np.argwhere(np.array(['Latency (clock cycles)' in line for line in lines])).flatten()[0] + 6]
      data_['latency_clks'] = int(lat_line.split('|')[2])
      data_['latency_ns']   = float(lat_line.split('|')[2])*5.0
      data_['latency_ii']   = int(lat_line.split('|')[4])
    
    return data_
  
  else:
    print('No synth reports found for {}! Returning empty data frame'.format(model_name))
    return data_
    
  
    
if __name__ == '__main__':
  
  parser = OptionParser()
  parser.add_option('-r','--reuse',action='store_true', dest='doReuse'  , default=False, help='Plot as function of reuse factor')
  parser.add_option('-p','--fromPkl',action='store_true', dest='fromPkl', default=False, help='Load performance metrics from pickle')
  parser.add_option('--aq',action='store_true', dest='addAQ', default=True, help='Add AutoQ')
  (options,args) = parser.parse_args()
  
  bitwidths = [16,14,12,10,8,6,4,3,2,1]
  reuse = [1,2,3,4,5,6]
  
  models = ['full','pruned_full','quant','pruned_quant']
  legends = ['BF','BP','Q','QP']
  # models = ['pruned_quant']
  # legends = ['QKeras Pruned']
  datas = []
  datasAQ = [] 
  aqnames = ['bayesian_v2_best_boosted_0','pruned_bayesian_v2_best_boosted_0']
  aqnames = ['latest_aq_0','pruned_latest_aq_0']
  aqlegs  = ['AQ','AQP']
  ## Do reuse scans
  if options.doReuse:
    colz = ['#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600']
    
    for model in models:
      data_scanreuse = {'w':[],'r':[], 'dsp':[], 'lut':[], 'ff':[],'bram':[],'dsp_rel':[], 'lut_rel':[], 'ff_rel':[],'bram_rel':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
      datas_r   = []
      legends_r = []
      for r in reuse:
        data = {'w':[],'r':[], 'dsp':[], 'lut':[], 'ff':[],'bram':[],'dsp_rel':[], 'lut_rel':[], 'ff_rel':[],'bram_rel':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
        for p in bitwidths:
          model_name = '{}_{}bit_reuse{}'.format(model,p,r)
          datai = getSynthData(model_name,p,r)
          for key in datai.keys():
            data[key].append(datai[key])
            if p == 16:
               data_scanreuse[key].append(datai[key])
        dataP = pandas.DataFrame(data)
        
        legends_r.append('R = {}'.format(int(r)))
        datas_r.append(dataP)
      dataR = pandas.DataFrame(data_scanreuse)
      datas.append(dataR)
      make_plots(datas_r,legends_r,outname='{}_per_reuse'.format(model),doAcc=False)
    print(datas)
    colz = ['#003f5c','#7a5195','#ef5675','#ffa600']
    make_2Dplots(datas,legends,outname='bw16'.format(model))
 
  #Do per-model bit width scans 
  else:
    if options.fromPkl:
      if options.addAQ:
        for aqname in aqnames:
            # data_ = getSynthData('{}_16bit_reuse1/'.format(aqname),p=0,r=1)
            #
            # model_ = tf.keras.models.load_model('models/{}/model_best.h5'.format(aqname),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
            # model_ = strip_pruning(model_)
            # (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=True)
            # del (x_train, y_train)
            # data_['accuracy_keras'], data_['accuracy_hls4ml'] = getQKeras(model_,'AutoQ',16,x_test, y_test)
            # print('Accuracy: Keras={} hls4ml={}'.format(data_['accuracy_keras'], data_['accuracy_hls4ml']))
            # print(data_)
            # dataAQ = pandas.DataFrame(data_, index=[0])
            # dataAQ.to_pickle(aqname+'.pkl')
            # datasAQ.append(dataAQ)
          dataAQ = pandas.read_pickle('{}.pkl'.format(aqname))
          print(aqname)
          print (dataAQ)
          datasAQ.append(dataAQ)
      for model_name in models:
        dataP = pandas.read_pickle('{}.pkl'.format(model_name))
        print(model_name)
        print (dataP)
        datas.append(dataP)
    else:
      (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=True)
      del (x_train, y_train)
      for model_name in models:
        model_baseline = tf.keras.models.load_model('models/{}.h5'.format(model_name),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
        # print(model_baseline.get_config())
        # sys.exit()
        score = model_baseline.evaluate(x_test, y_test)
        print('Keras Accuracy {} = {}'.format(model_name,score[1]))
        model_baseline_stripped = strip_pruning(model_baseline)
        intbits_a = 0
        intbits_w = 0
        if model_name.find('full')!=-1:
          a = hls4ml.model.profiling.activations_keras(model_baseline_stripped, x_test[:100], fmt='summary')
          intbits_a = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], a)))))) + 1)
          w = hls4ml.model.profiling.weights_keras(model_baseline_stripped, fmt='summary')
          intbits_w = int(np.ceil(max(np.log2(np.array(list(map(lambda x : x['whishi'], w)))))) + 1)

        data = {'w':[],'accuracy_keras':[],'accuracy_hls4ml':[],  'dsp':[], 'lut':[], 'ff':[],'bram':[],'dsp_rel':[], 'lut_rel':[], 'ff_rel':[],'bram_rel':[], 'latency_clks':[], 'latency_ns':[], 'latency_ii':[]}
        for p in bitwidths:
          print('For model {} and precision {}'.format(model_name, p))
          datai = getReports(model_name,p,x_test, y_test)
          for key in datai.keys():
            data[key].append(datai[key])
        dataP = pandas.DataFrame(data)
        dataP.to_pickle(model_name+'.pkl')
        datas.append(dataP)
      if options.addAQ:
        for aqname in aqnames:
          data_ = getSynthData('{}_16bit_reuse1/'.format(aqname),p=0,r=1)
        
          model_ = tf.keras.models.load_model('models/{}/model_best.h5'.format(aqname),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
          model_ = strip_pruning(model_)
          (x_train, y_train), (x_test, y_test) = getNumpyData('svhn_cropped',oneHot=True)
          del (x_train, y_train)
          data_['accuracy_keras'], data_['accuracy_hls4ml'] = getQKeras(model_,'AutoQ',16,x_test, y_test)
          print('Accuracy: Keras={} hls4ml={}'.format(data_['accuracy_keras'], data_['accuracy_hls4ml']))
          print(data_)
          dataAQ = pandas.DataFrame(data_, index=[0])
          dataAQ.to_pickle(aqname+'.pkl')
          datasAQ.append(dataAQ)
    if options.addAQ:
      print('Making AQ plots!!')
      make_plots(datas,legends,outname='perModel_wAQ',doAcc=True,datas_AQ=datasAQ,legs_AQ=aqlegs)
    else:
      make_plots(datas,legends,outname='perModel',doAcc=True)
