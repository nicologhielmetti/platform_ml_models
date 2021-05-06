import os
import setGPU
# edit depending on where Vivado is installed:
# os.environ['PATH'] = '/<Xilinx installation directory>/Vivado/<version>/bin:' + os.environ['PATH']
os.environ['PATH'] = '/xilinx/Vivado/2019.2/bin:' + os.environ['PATH']
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from train import yaml_load
from tensorflow.keras.datasets import cifar10
import argparse
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))
            
def main(args):
    # parameters
    our_config = yaml_load(args.config)
    save_dir = our_config['save_dir']
    model_name = our_config['model']['name']
    model_file_path = os.path.join(save_dir, 'model_best.h5')

    from tensorflow.keras.models import load_model
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)    

    model = load_model(model_file_path, custom_objects=co)
    if bool(our_config['convert']['RemoveSoftmax']):
        input_layer = None
        output_layer = None
        for layer in model.layers:
            if layer.name == 'q_conv2d_batchnorm_1': 
                input_layer = Input(layer.input.shape[1:],name='input_1')
                output_layer = layer(input_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.save(model_file_path.replace('.h5','_1layer.h5'))
        
    model.summary()
    tf.keras.utils.plot_model(model,
                              to_file="model.png",
                              show_shapes=True,
                              show_dtype=False,
                              show_layer_names=False,
                              rankdir="TB",
                              expand_nested=False)

    import hls4ml
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')

    print("-----------------------------------")
    print_dict(config)
    print("-----------------------------------")

    config['Model'] = {}
    config['Model']['ReuseFactor'] = our_config['convert']['ReuseFactor']
    config['Model']['Strategy'] = our_config['convert']['Strategy']
    config['Model']['Precision'] = our_config['convert']['Precision']
    for name in config['LayerName'].keys():
        config['LayerName'][name]['Trace'] = bool(our_config['convert']['Trace'])
        config['LayerName'][name]['ReuseFactor'] = our_config['convert']['ReuseFactor']
        config['LayerName'][name]['Precision'] = our_config['convert']['Precision']
        if 'activation' in name:
            config['LayerName'][name]['Precision'] = {}
            config['LayerName'][name]['Precision']['result'] = our_config['convert']['PrecisionActivation']
    # custom configs
    for name in our_config['convert']['Override'].keys():
        config['LayerName'][name].update(our_config['convert']['Override'][name])

    cfg = hls4ml.converters.create_backend_config(fpga_part='xc7z020clg400-1')
    cfg['HLSConfig'] = config
    cfg['IOType'] = our_config['convert']['IOType']
    cfg['Backend'] = our_config['convert']['Backend']
    cfg['InputData'] = 'input_data.dat'
    cfg['OutputPredictions'] = 'output_predictions.dat'
    cfg['Interface'] = 's_axilite' # or 'm_axi'
    cfg['ClockPeriod'] = our_config['convert']['ClockPeriod']
    cfg['KerasModel'] = model
    cfg['OutputDir'] = our_config['convert']['OutputDir']

    print("-----------------------------------")
    print_dict(cfg)
    print("-----------------------------------")

    # profiling / testing
    hls_model = hls4ml.converters.keras_to_hls(cfg)

    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='model_hls4ml.png')

    # Bitfile time
    if bool(our_config['convert']['Build']):
        hls_model.build(reset=True,csim=False,cosim=False,validation=False,synth=True,vsynth=False,export=False)
        hls4ml.report.read_vivado_report(our_config['convert']['OutputDir'])
        if our_config['convert']['Backend'] == 'Pynq':
            hls4ml.templates.PynqBackend.make_bitfile(hls_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = "baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)
