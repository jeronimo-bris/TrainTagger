"""DeepSet model child class

Written 28/05/2025 cebrown@cern.ch
"""

import json
import os

import hls4ml
import tensorflow as tf
from schema import Schema, And

from tagger.model.common_tensorflow import choose_aggregator, initialise_tensorflow
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.HGQ2Model import HGQ2Model

from tensorflow.keras.layers import Activation, BatchNormalization
from hgq.layers import QConv1D, QDense

from tagger.model.common_hgq import hgq_quantization_decorator
# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('DeepSetModelHGQ2')
class DeepSetModelHGQ2(HGQ2Model):

    """DeepSetModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "classification_layers" : list,
                                  "regression_layers" : list,
                                  "kernel_initializer" : str,
                                  "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"])},
                "quantization_config" : HGQ2Model.quantization_schema,
                "training_config" : HGQ2Model.training_config_schema,
                "autoencoder_config" : HGQ2Model.autoencoder_config_schema,
                ## generic hls4ml configuration
                "firmware_config" : {"input_precision" : dict,
                                    "class_precision" : str,
                                    "reg_precision": str,
                                    "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                    "fpga_part" : str,
                                    "project_name" : str},
                "inputs" : {
                    "basic_input_config": list,
                    "basic_features": list,
                    "jet_features": list}
            }
    )

    @hgq_quantization_decorator
    def build_model(self, inputs_shape: dict, outputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
            outputs_shape (tuple): Shape of the output

        Additional hyperparameters in the config
            conv1d_layers: List of number of nodes for each layer of the conv1d layers.
            classifier_layers: List of number of nodes for each layer of the classifier MLP.
            regression_layers: List of number of nodes for each layer of the regression MLP
            aggregator: String that specifies the type of aggregator to use after the conv1D net.
        """

        initialise_tensorflow(self.run_config['num_threads'])

        self.common_args = {
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        # Initialize inputs
        inputs = tf.keras.layers.Input(shape=inputs_shape['basic_input'], name='basic_input')

        # Main branch
        main = BatchNormalization(name='norm_input')(inputs)

        # Make Conv1D layers
        for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
            main = QConv1D(filters=depthconv1d, kernel_size=1, activation='relu', name='Conv1D_' + str(iconv1d + 1), **self.common_args)(main)

        agg = choose_aggregator(choice=self.model_config['aggregator'], name="pool")
        main = agg(main)

        # Now split into jet ID and pt regression

        # Make fully connected dense layers for classification task
        jet_id = main
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            jet_id = QDense(depthclass, activation="relu", name='Dense_' + str(iclass + 1) + '_jetID', **self.common_args)(jet_id)

        # Make output layer for classification task
        jet_id = QDense(outputs_shape[0], name='Dense_' + str(iclass + 2) + '_jetID', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)

        # Make fully connected dense layers for pt regression task
        pt_regress = main
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            pt_regress = QDense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', **self.common_args)(pt_regress)

        bits, integer_bits = self.quantization_config['pt_output_quantization']
        pt_regress = QDense(1, name='pT_output')(pt_regress)

        # Define the model using both branches
        self.jet_model = tf.keras.Model(inputs=inputs, outputs={"jet_id_output": jet_id, "pT_output": pt_regress})
        self.encoder_model = tf.keras.Model(inputs=inputs, outputs=main, name="encoder")

        print(self.jet_model.summary())


    def firmware_convert(self, firmware_dir: str, build: bool = False):
        """Run the hls4ml model conversion

        Args:
            firmware_dir (str): Where to save the firmware
            build (bool, optional): Run the full hls4ml build? Or just create the project. Defaults to False.
        """

        # Remove the old directory if it exists
        hls4ml_outdir = firmware_dir + '/' + self.firmware_config['project_name']
        os.system(f'rm -rf {hls4ml_outdir}')

        # Create default config
        config = hls4ml.utils.config_from_keras_model(self.jet_model, granularity='name')
        config['IOType'] = 'io_parallel'
        for layer in self.firmware_config['input_precision']:
            config['LayerName'][layer]['Precision']['result'] = self.firmware_config['input_precision'][layer]

        # Check that the input layers in the model match the input_precision keys in the firmware config
        input_layer_names = [layer.name for layer in self.jet_model.layers if isinstance(layer, tf.keras.layers.InputLayer)]
        assert set(input_layer_names) == set(self.firmware_config['input_precision'].keys())

        # Configuration for conv1d layers
        # hls4ml does not !!! automatically figure out the paralellization factor, this leads to csim, hdl sim errors
        config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 16
        config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 16

        # Additional config
        for layer in self.jet_model.layers:
            layer_name = layer.__class__.__name__
            if layer_name in ["BatchNormalization", "InputLayer"]:
                for k in self.firmware_config['input_precision'].keys():
                    if k in layer.name:
                        precision = self.firmware_config['input_precision'][k]
                        break  # stop once we found a match
                config["LayerName"][layer.name]["Precision"] = precision
                config["LayerName"][layer.name]["result"] = precision
                config["LayerName"][layer.name]["Trace"] = not build

            elif layer_name in ["Permute", "Concatenate", "Flatten", "Reshape", "UpSampling1D", "Add"]:
                print("Skipping trace for:", layer.name)
            else:
                config["LayerName"][layer.name]["Trace"] = not build

        config["LayerName"]["jet_id_output"]["Precision"]["result"] = self.firmware_config['class_precision']
        config["LayerName"]["jet_id_output"]["Implementation"] = "stable"
        config["LayerName"]["pT_output"]["Precision"]["result"] = self.firmware_config['reg_precision']
        config["LayerName"]["pT_output"]["Implementation"] = "latency"

        # Write HLS
        self.hls_jet_model = hls4ml.converters.convert_from_keras_model(
            self.jet_model,
            backend='Vitis',
            project_name=self.firmware_config['project_name'],
            clock_period=self.firmware_config['clock_period'],
            hls_config=config,
            output_dir=f'{hls4ml_outdir}',
            part= self.firmware_config['fpga_part'],
        )

        # Compile the project
        self.hls_jet_model.compile()

        # Save config  as json file
        print("Saving default config as config.json ...")
        with open(hls4ml_outdir + '/config.json', 'w') as fp:
            json.dump(config, fp)

        old_text = '#include <tuple>'
        new_text = ""

        # manually remove the #include <tuple>
        path = hls4ml_outdir + '/firmware/defines.h'
        with open(path, 'r') as f:
            content = f.read()

        content = content.replace('#include <tuple>', '')

        with open(path, 'w') as f:
            f.write(content)

        if build:
            # build the project
            self.hls_jet_model.build(csim=False, reset=True)
