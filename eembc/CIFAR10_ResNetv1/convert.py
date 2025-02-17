from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
import argparse
from tensorflow.keras.datasets import cifar10
from train import yaml_load
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from tensorflow.keras.utils import to_categorical
import numpy as np
import hls4ml
from qkeras.utils import _add_supported_quantized_objects
import tensorflow as tf
import os
import setGPU
# edit depending on where Vivado is installed:
# os.environ['PATH'] = '/<Xilinx installation directory>/Vivado/<version>/bin:' + os.environ['PATH']
os.environ['PATH'] = '/xilinx/Vivado/2019.1/bin:' + os.environ['PATH']


def print_dict(d, indent=0):
    align = 20
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
        input_layer = model.inputs
        output_layer = None
        for layer in model.layers:
            if layer.name == 'softmax':
                output_layer = layer.input
        model = Model(inputs=input_layer, outputs=output_layer)
        model.save(model_file_path.replace('.h5', '_nosoftmax.h5'))

    model.summary()
    tf.keras.utils.plot_model(model,
                              to_file="model.png",
                              show_shapes=True,
                              show_dtype=False,
                              show_layer_names=False,
                              rankdir="TB",
                              expand_nested=False)

    _, (X_test, y_test) = cifar10.load_data()
    X_test = np.ascontiguousarray(X_test/256.)
    num_classes = 10
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # just use first 100
    #if bool(our_config['convert']['Trace']):
    if True:
        X_test = X_test[:100]
        y_test = y_test[:100]

    y_keras = model.predict(X_test)
    print("Keras Accuracy:  {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))

    np.save('y_keras.npy', y_keras)
    np.save('y_test.npy', y_test)
    np.save('X_test.npy', X_test)

    import hls4ml
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')

    print("-----------------------------------")
    print_dict(config)
    print("-----------------------------------")

    #config['Model'] = {}
    config['Model']['ReuseFactor'] = our_config['convert']['ReuseFactor']
    config['Model']['Strategy'] = our_config['convert']['Strategy']
    config['Model']['Precision'] = our_config['convert']['Precision']
    config['SkipOptimizers'] = ['reshape_stream']
    print(config['LayerName'].keys())
    for name in config['LayerName'].keys():
        config['LayerName'][name]['Trace'] = bool(our_config['convert']['Trace'])
        config['LayerName'][name]['ReuseFactor'] = our_config['convert']['ReuseFactor']
        config['LayerName'][name]['Precision'] = our_config['convert']['Precision']
    # custom configs
    for name in our_config['convert']['Override'].keys():
        if name not in config['LayerName'].keys():
            config['LayerName'][name] = {}
        config['LayerName'][name].update(our_config['convert']['Override'][name])


    backend = our_config['convert']['Backend']
    clock_period = our_config['convert']['ClockPeriod']
    io_type = our_config['convert']['IOType']
    interface = our_config['convert']['Interface']
    if backend == 'VivadoAccelerator':
        board = our_config['convert']['Board']
        driver = our_config['convert']['Driver']
        cfg = hls4ml.converters.create_config(backend=backend, board=board, interface=interface, clock_period=clock_period,
                                              io_type=io_type, driver=driver)
    else:
        part = our_config['convert']['XilinxPart']
        cfg = hls4ml.converters.create_config(backend=backend, part=part, clock_period=clock_period,
                                              io_type=io_type)
    cfg['HLSConfig'] = config
    cfg['InputData'] = 'input_data.dat'
    cfg['OutputPredictions'] = 'output_predictions.dat'
    cfg['KerasModel'] = model
    cfg['OutputDir'] = our_config['convert']['OutputDir']

    print("-----------------------------------")
    print_dict(cfg)
    print("-----------------------------------")

    # profiling / testing
    hls_model = hls4ml.converters.keras_to_hls(cfg)

    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='model_hls4ml.png')

    if bool(our_config['convert']['Trace']):
        from hls4ml.model.profiling import compare, numerical
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure()
        wp, wph, ap, aph = numerical(model=model, hls_model=hls_model, X=X_test)
        plt.show()
        plt.savefig('profiling_numerical.png', dpi=300)

        #plt.figure()
        #cp = compare(keras_model=model, hls_model=hls_model, X=X_test, plot_type="dist_diff")
        #plt.show()
        #plt.savefig('profiling_compare.png', dpi=300)

        y_hls, hls4ml_trace = hls_model.trace(X_test)
        np.save('y_hls.npy', y_hls)
        keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_test)

        for layer in hls4ml_trace.keys():
            plt.figure()
            klayer = layer
            if '_alpha' in layer:
                klayer = layer.replace('_alpha', '')
            plt.scatter(hls4ml_trace[layer].flatten(), keras_trace[klayer].flatten(), s=0.2)
            min_x = min(np.amin(hls4ml_trace[layer]), np.amin(keras_trace[klayer]))
            max_x = max(np.amax(hls4ml_trace[layer]), np.amax(keras_trace[klayer]))
            plt.plot([min_x, max_x], [min_x, max_x], c='gray')
            plt.xlabel('hls4ml {}'.format(layer))
            plt.ylabel('QKeras {}'.format(klayer))
            plt.show()
            plt.savefig('profiling_{}.png'.format(layer), dpi=300)
    else:
        hls_model.compile()
        y_hls = hls_model.predict(X_test)

    print("Keras Accuracy:  {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
    print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))

    # Bitfile time
    if bool(our_config['convert']['Build']):
        np.savetxt('input_data.dat', X_test[:1].reshape(1, -1), fmt='%f', delimiter=' ')
        np.savetxt('output_predictions.dat', y_keras[:1].reshape(1, -1), fmt='%f', delimiter=' ')
        hls_model.build(reset=False, csim=True, cosim=True, validation=True, synth=True, vsynth=True, export=True)
        hls4ml.report.read_vivado_report(our_config['convert']['OutputDir'])
        if our_config['convert']['Backend'] == 'VivadoAccelerator':
            hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)
