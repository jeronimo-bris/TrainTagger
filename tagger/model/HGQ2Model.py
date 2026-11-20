"""QKeras model parent class

Written 29/09/2025 cebrown@cern.ch
"""

import os
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from schema import And, Optional

from keras import Model
from keras.models import load_model

# Qkeras
from hgq.utils.sugar import FreeEBOPs, BetaPID

from tagger.model.common_hgq import ReduceLROnPlateauWithEbopsThres, EarlyStoppingWithEbopsThres
from tagger.model.common_tensorflow import huber_loss
from tagger.model.JetTagModel import JetTagModel
from tagger.model.AutoencoderModelHGQ2 import AutoencoderModelHGQ2, EncoderMetricsCallback
from tagger.data.tools import constituents_mask

class HGQ2Model(JetTagModel):
    """HGQ2Model class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    autoencoder_config_schema = {'layers': And(
                              list,
                              lambda layers: len(layers) % 2 == 1,
                              lambda layers: layers == layers[::-1],
                              lambda layers: all(isinstance(units, int) and units > 0 for units in layers),
                          ),
                          "l2_normalize": bool,
                          "epochs" : And(int, lambda s: s >= 1),
                          "batch_size" : And(int, lambda s: s >= 1),
                          "learning_rate" : And(float, lambda s: s > 0.0),
                          "validation_split" : And(float, lambda s: 0.0 < s < 1.0),
                          "target_ebops": And(int, lambda s: s > 0),
                          "EarlyStopping_patience" : And(int, lambda s: s > 0),
                          "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                          "ReduceLROnPlateau_patience" : int,
                          "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0),
                          }

    quantization_schema = {'quantizer_bits' : And(int, lambda s: 64 >= s >= 0),
                           'quantizer_bits_int' : And(int, lambda s: 32 >= s >= 0),
                           'reg_quantizer_bits' : And(int, lambda s: 64 >= s >= 0),
                           'reg_quantizer_bits_int' : And(int, lambda s: 32 >= s >= 0),
                           'quantizer_alpha_val' : And(float, lambda s: 1.0 >= s >= 0.0),
                           'pt_output_quantization' : list,
                           # 'pt_layers_bits': list,
                           }

    training_config_schema =    {"weight_method": And(
                                        list,
                                        lambda lst: len(lst) == 2,
                                        lambda lst: all(x in ["none", "ptref", "onlyclass"] for x in lst)
                                    ),
                                 "validation_split" : And(float, lambda s: s > 0.0),
                                 "epochs" : And(int, lambda s: s >= 1),
                                 "batch_size" : And(int, lambda s: s >= 1),
                                 "learning_rate" : And(float, lambda s: s > 0.0),
                                 "loss_weights" : And(list, lambda s: len(s) == 2),
                                 "target_ebops": And(int, lambda s: s > 0),
                                 "EarlyStopping_patience" : And(int, lambda s: s > 0),
                                 "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                                 "ReduceLROnPlateau_patience" : int,
                                 "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0),
                                 "pileup": And(bool),
                                 "huber_weights": And(list, lambda s: len(s) == 2),
                                 "huber_delta": And(float, lambda s: s > 0.0),
                                 Optional("embedding_monitor"): {
                                     "enabled": bool,
                                     "sample_size": And(int, lambda s: s >= 2),
                                     "batch_size": And(int, lambda s: s >= 1),
                                     "every_n_epochs": And(int, lambda s: s >= 1),
                                 },
                                 }


    def compile_model(self, num_samples: int):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            BetaPID(
                p=1, i=0.1, d=0,
                target_ebops=self.training_config['target_ebops'],
                init_beta=1e-10, warmup=10, max_beta=1e-6, damp_beta_on_target=0.5
            ),
            EarlyStoppingWithEbopsThres(
                monitor="val_loss",
                patience=self.training_config['EarlyStopping_patience'],
                mode="min",
                restore_best_weights=True,
                ebops_threshold=self.training_config['target_ebops'],
                verbose=2,
            ),
            ReduceLROnPlateauWithEbopsThres(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
                ebops_threshold=self.training_config['target_ebops'],
                verbose=2,
            ),
            FreeEBOPs(),
        ]

        # compile the tensorflow model setting the loss and metrics
        self.jet_model.compile(
            optimizer='adam',
            loss={
                self.loss_name + self.output_id_name: 'categorical_crossentropy',
                self.loss_name + self.output_pt_name: huber_loss(
                    delta=self.training_config['huber_delta'],
                    pu=self.training_config['huber_weights'][0],
                    alpha=self.training_config['huber_weights'][1]),
            },
            loss_weights=self.training_config['loss_weights'],
            metrics={
                self.loss_name + self.output_id_name: 'categorical_accuracy',
                self.loss_name + self.output_pt_name: ['mae', 'mean_squared_error'],
            },
            weighted_metrics={
                self.loss_name + self.output_id_name: 'categorical_accuracy',
                self.loss_name + self.output_pt_name: ['mae', 'mean_squared_error'],
            },
        )

    def fit(
        self,
        inputs_dict: dict,
        y_train: npt.NDArray[np.float64],
        pt_target_train: npt.NDArray[np.float64],
        sample_weight: [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ):
        """Fit the model to the training dataset

        Args:
            inputs_dict (npt.NDArray[np.float64]): X train dataset, containts inputs and pt
            y_train (npt.NDArray[np.float64]): y train classification targets
            pt_target_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """

        callbacks = list(self.callbacks)
        monitor_config = self.training_config.get("embedding_monitor", {})

        if monitor_config.get("enabled", False):
            # Keras takes the final fraction of array inputs as its validation
            # set. Use a fixed subset of that same region as the monitoring
            # probe so epoch-to-epoch changes reflect the encoder, not sampling.
            x_train = inputs_dict['basic_input']
            validation_start = int(
                len(x_train) * (1.0 - self.training_config['validation_split'])
            )
            x_probe = x_train[validation_start:]
            y_probe = y_train[validation_start:]

            sample_size = min(monitor_config['sample_size'], len(x_probe))
            if sample_size < 2:
                raise ValueError(
                    "Embedding monitoring requires at least two validation samples"
                )
            if sample_size < len(x_probe):
                probe_indices = np.linspace(
                    0,
                    len(x_probe) - 1,
                    num=sample_size,
                    dtype=int,
                )
                x_probe = x_probe[probe_indices]
                y_probe = y_probe[probe_indices]

            autoencoder_layers = self.autoencoder_config.get("layers", [])
            pca_rank = (
                autoencoder_layers[len(autoencoder_layers) // 2]
                if autoencoder_layers
                else None
            )

            self.embedding_metrics_callback = EncoderMetricsCallback(
                x_probe=x_probe,
                y_probe=y_probe,
                pca_rank=pca_rank,
                layer_name="pool",
                batch_size=monitor_config['batch_size'],
                every_n_epochs=monitor_config['every_n_epochs'],
                csv_path=os.path.join(
                    self.output_directory,
                    "encoder_embedding_metrics.csv",
                ),
            )
            callbacks.append(self.embedding_metrics_callback)

        # Train the model using hyperparameters in yaml config
        self.history = self.jet_model.fit(
            inputs_dict['basic_input'],
            {self.loss_name + self.output_id_name: y_train, self.loss_name + self.output_pt_name: pt_target_train},
            sample_weight={
                'jet_id_output': sample_weight[0],
                'pT_output': sample_weight[1],
            },
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=callbacks,
            shuffle=True,
        )


    def prepare_inputs(self, raw_inputs: dict) -> dict:
        """Prepare the input dictionary for the model from a list of arrays

        Args:
            raw_inputs: Dictionary of all possible input arrays (currently requires basic_input, jet_pt and jet_eta)

        Returns:
            dict: Dictionary of required input arrays
        """

        # get relevant feature indices
        pt_rel_idx = self.particle_input_vars.index("pt_rel")

        # build all possible inputs, add here if ever in need of new ones
        input_dict = {
            'basic_input': raw_inputs['basic_input'],
            'basic_mask': constituents_mask(raw_inputs['basic_input'], 10),
            'pt_mask': constituents_mask(raw_inputs['basic_input'], 10)[:, :, 0],
            'constituent_fraction': raw_inputs['basic_input'][:, :, pt_rel_idx],
        }

        # remove unused inputs
        for key in list(input_dict.keys()):
            if key not in self.inputs['basic_features']:
                del input_dict[key]

        # add jet features if specified in model config
        if len(self.inputs['jet_features']) > 0:
            input_dict['jet_features'] = raw_inputs['jet_features']

        input_shapes = {k: v.shape[1:] for k, v in input_dict.items()}

        return input_dict, input_shapes

    def get_keras_trace_model(self):
        # Create a sub-model that outputs all intermediate layers and get keras trace
        layer_outputs = [layer.output for layer in self.jet_model.layers]
        keras_trace_model = Model(inputs=self.jet_model.input, outputs=layer_outputs)
        return keras_trace_model

    @staticmethod
    def _get_branch_inputs(output_tensor):
        """
        Return only keras Input tensors that ACTUALLY feed into output_tensor.
        """
        visited = set()
        inputs = {}
        stack = [output_tensor]

        while stack:
            t = stack.pop()
            key = t.ref()
            if key in visited:
                continue
            visited.add(key)

            kh = t._keras_history
            layer = kh.layer
            node_index = kh.node_index

            # If this tensor comes from an InputLayer
            if isinstance(layer, tf.keras.layers.InputLayer):
                inputs[layer.name] = layer.output
                continue

            # Follow ONLY the node that produced this tensor
            node = layer._inbound_nodes[node_index]
            inbound_tensors = tf.nest.flatten(node.input_tensors)
            stack.extend(inbound_tensors)

        return list(inputs.values())

    def get_branch_model(self, branch):
        """
        Build a standalone sub-model for one output branch of self.jet_model,
        plus the matching ordered list of input arrays from test_dict.
        """
        output_tensor = self.jet_model.get_layer(branch).output
        input_layers = self._get_branch_inputs(output_tensor)
        branch_model = tf.keras.Model(input_layers, output_tensor)

        return branch_model

    # Decorated with save decorator for added functionality
    @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """

        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        self.jet_model.save(os.path.join(out_dir, "model/saved_model.keras"))
        self.autoencoder_model.save(os.path.join(out_dir, "model/saved_autoencoder.keras"))
        print(f"Model saved to {os.path.join(out_dir, 'model')}")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """

        # # Additional custom objects for attention layers
        # custom_objects_ = {
        #     "AAtt": AAtt,
        #     "AttentionPooling": AttentionPooling,
        # }

        # Load the model
        self.jet_model = load_model(
            f"{out_dir}/model/saved_model.keras",
            # custom_objects=custom_objects_,
            compile=False,
        )
        self.encoder_model = tf.keras.Model(self.jet_model.input, self.jet_model.get_layer('pool').output)
        self.configure_autoencoder()
        self.autoencoder_model.load(f"{out_dir}/model/saved_autoencoder.keras")


    def configure_autoencoder(self):
        """Attach an independent autoencoder to this model."""

        if self.encoder_model is None:
            raise RuntimeError("Build the DeepSets model first.")

        embedding_dim = int(self.encoder_model.output_shape[-1])
        self.autoencoder_model = AutoencoderModelHGQ2(
            input_dim=embedding_dim,
            config=self.autoencoder_config,
        )

        return self.autoencoder_model

    def fit_autoencoder(
        self,
        inputs_dict: dict,
        sample_weight: [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ):

        if self.autoencoder_model is None:
            raise RuntimeError("Call configure_autoencoder() first.")

        z = self.embed(inputs_dict, batch_size=20_000)
        self.autoencoder_history = self.autoencoder_model.fit(
            x=z,
            sample_weight=sample_weight[0],
            metrics_csv_path=os.path.join(
                self.output_directory,
                "autoencoder_metrics.csv",
            ),
            epochs=self.autoencoder_config['epochs'],
            batch_size=self.autoencoder_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.autoencoder_config['validation_split'],
            shuffle=True,
        )

        return self.autoencoder_history
