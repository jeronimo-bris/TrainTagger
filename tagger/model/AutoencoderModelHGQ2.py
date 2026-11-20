import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from hgq.layers import QDense
from hgq.utils.sugar import FreeEBOPs, BetaPID

from tagger.model.common_hgq import (
    EarlyStoppingWithEbopsThres,
    ReduceLROnPlateauWithEbopsThres,
    hgq_quantization_decorator,
)



class AutoencoderModelHGQ2:

    def __init__(self, input_dim: int, config: dict):

        self.input_dim = input_dim
        self.model_layers = config['layers']
        self.l2_normalize = config['l2_normalize']
        self.learning_rate = config['learning_rate']
        self.target_ebops = config['target_ebops']
        self.config = config

        if len(self.model_layers) % 2 == 0:
            raise ValueError("Autoencoder layers must contain one central bottleneck layer.")
        if self.model_layers != self.model_layers[::-1]:
            raise ValueError("Autoencoder layers must be symmetric around the bottleneck.")

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()

        self.build_model(input_shape=(input_dim,), output_shape=(input_dim,))
        self.compile()


    @hgq_quantization_decorator
    def build_model(self, input_shape, output_shape):

        num_layers = len(self.model_layers)
        inputs = keras.layers.Input(shape=input_shape, name="input_layer")

        x = inputs
        for i, units in enumerate(self.model_layers[: num_layers // 2]):
            x = QDense(units=units, activation="relu", parallelization_factor=units, name=f"encoder_dense_{i}")(x)

        x = QDense(units=self.model_layers[num_layers // 2], activation=None, parallelization_factor=self.model_layers[num_layers // 2], name="bottleneck")(x)

        for i, units in enumerate(self.model_layers[num_layers // 2 + 1 :]):
            x = QDense(units=units, activation="relu", parallelization_factor=units, name=f"decoder_dense_{i}")(x)

        x = QDense(units=output_shape[0], activation=None, parallelization_factor=output_shape[0], name="reconstruction")(x)

        self.model = keras.Model(inputs=inputs, outputs=x, name="autoencoder")

    def compile(self, optimizer=None, loss=None, **kwargs):
        optimizer = optimizer or self.optimizer
        loss = loss or self.loss_fn
        self.model.compile(optimizer=optimizer, loss=loss, **kwargs)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def fit(
            self, 
            x: np.ndarray,
            callbacks=None,
            metrics_csv_path="autoencoder_metrics", 
            **kwargs,
        ):

        if self.l2_normalize:
            x = tf.math.l2_normalize(x, axis=1)

        if callbacks is None:
            callbacks = [
                BetaPID(
                    p=1, i=0.1, d=0,
                    target_ebops=self.target_ebops,
                    init_beta=1e-10, warmup=10,
                    max_beta=1e-6, damp_beta_on_target=0.5
                ),
                EarlyStoppingWithEbopsThres(
                    monitor="val_loss",
                    patience=self.config['EarlyStopping_patience'],
                    verbose=1,
                    mode="min",
                    restore_best_weights=True,
                    ebops_threshold=self.target_ebops,
                ),
                ReduceLROnPlateauWithEbopsThres(
                    monitor='val_loss',
                    factor=self.config['ReduceLROnPlateau_factor'],
                    patience=self.config['ReduceLROnPlateau_patience'],
                    min_lr=self.config['ReduceLROnPlateau_min_lr'],
                    ebops_threshold=self.target_ebops,
                ),
                FreeEBOPs(),
                AutoencoderMetricsCallback(
                    x_probe=x,
                    batch_size=10_000,
                    every_n_epochs=1,
                    csv_path=metrics_csv_path,
                )
            ]

        self.history = self.model.fit(
            x,
            y=x,
            callbacks=callbacks,
            **kwargs,
        )
        return self.history

    def save(self, path: str):
        """Save the wrapped Keras autoencoder."""
        self.model.save(path)

    def load(self, path: str):
        """Restore the wrapped Keras autoencoder without changing this API."""
        self.model = keras.models.load_model(path, compile=False)
        self.compile()
        return self

    def score(self, x: np.ndarray, batch_size: int = 20_000, verbose: int = 1,) -> np.ndarray:

        if self.l2_normalize:
            x = tf.math.l2_normalize(x, axis=1)

        x_recon = self.model.predict(x, batch_size=batch_size, verbose=verbose,)
        return np.sqrt(np.mean((x - x_recon) ** 2, axis=1,))






def spectrum_metrics(
    x: np.ndarray,
    explained_variance_rank: int | None = None,
) -> dict:
    """Compute metrics from the centered covariance spectrum.

    x: (n_samples, n_features)
    """
    if explained_variance_rank is not None and explained_variance_rank < 1:
        raise ValueError("explained_variance_rank must be positive")

    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("Expected at least two 2D samples")

    x = x - x.mean(axis=0, keepdims=True)

    # Singular values squared are proportional to
    # eigenvalues of the covariance matrix.
    s = np.linalg.svd(x, compute_uv=False)
    eigenvalues = s**2

    total = eigenvalues.sum()

    if total < 1e-20:
        metrics = {
            "effective_rank": 0.0,
            "effective_rank_fraction": 0.0,
            "participation_ratio": 0.0,
            "pca95": 0,
            "total_variance": 0.0,
        }
        if explained_variance_rank is not None:
            metrics[
                f"pca{explained_variance_rank}_explained_variance"
            ] = 0.0
        return metrics

    p = eigenvalues / total
    p_nonzero = p[p > 0]

    # Shannon entropy effective rank
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    effective_rank = np.exp(entropy)

    # Participation ratio (inverse concentration)
    participation_ratio = 1.0 / np.sum(p**2)

    # Number of PCs explaining 95% of total variance
    pca95 = np.searchsorted(np.cumsum(p), 0.95) + 1

    max_rank = min(x.shape[0] - 1, x.shape[1])

    metrics = {
        "effective_rank": float(effective_rank),
        "effective_rank_fraction": float(effective_rank / max_rank),
        "participation_ratio": float(participation_ratio),
        "pca95": int(pca95),
        "total_variance": float(total / (x.shape[0] - 1)),
    }
    if explained_variance_rank is not None:
        rank = min(explained_variance_rank, len(p))
        metrics[
            f"pca{explained_variance_rank}_explained_variance"
        ] = float(np.sum(p[:rank]))

    return metrics


def effective_rank(x: np.ndarray) -> float:
    return spectrum_metrics(x)["effective_rank"]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA similarity between two representations."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = (np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro"))

    if denominator < 1e-20:
        return 0.0

    return float(numerator / denominator)


def fisher_separation_ratio(x: np.ndarray, labels: np.ndarray) -> float:
    """Trace ratio of between-class to within-class scatter."""

    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)

    if x.ndim != 2 or labels.ndim != 1 or len(x) != len(labels):
        raise ValueError("Expected aligned 2D embeddings and 1D class labels")

    classes = np.unique(labels)
    if len(classes) < 2:
        return 0.0

    global_mean = x.mean(axis=0)
    between_class_scatter = 0.0
    within_class_scatter = 0.0

    for class_label in classes:
        class_embedding = x[labels == class_label]
        class_mean = class_embedding.mean(axis=0)
        between_class_scatter += len(class_embedding) * np.sum(
            (class_mean - global_mean) ** 2
        )
        within_class_scatter += np.sum((class_embedding - class_mean) ** 2)

    return float(between_class_scatter / max(within_class_scatter, 1e-20))


def reconstruction_metrics(x, x_recon) -> dict:
    """Measure reconstruction fidelity."""

    x = np.asarray(x, dtype=np.float64)
    x_recon = np.asarray(x_recon, dtype=np.float64)

    residual = x - x_recon

    mse = np.mean(residual**2)

    # Global R2 relative to the mean input embedding
    total_ss = np.sum((x - x.mean(axis=0, keepdims=True)) ** 2)
    residual_ss = np.sum(residual**2)
    r2 = 1.0 - residual_ss / max(total_ss, 1e-20)

    # Mean sample-wise cosine similarity
    dot = np.sum(x * x_recon, axis=1)
    norms = (np.linalg.norm(x, axis=1) * np.linalg.norm(x_recon, axis=1))
    cosine = np.mean(dot / np.maximum(norms, 1e-12))

    # Retention of total centered representation variance
    input_var = np.sum(np.var(x, axis=0))
    recon_var = np.sum(np.var(x_recon, axis=0))
    variance_retention = recon_var / max(input_var, 1e-20)

    return {
        "mse": float(mse),
        "r2": float(r2),
        "cosine": float(cosine),
        "variance_retention": float(variance_retention),
    }

# Custom monitoring callback
class AutoencoderMetricsCallback(keras.callbacks.Callback):
    def __init__(
        self,
        x_probe: np.ndarray,
        batch_size: int = 256,
        every_n_epochs: int = 1,
        csv_path: str | None = None,
    ):
        super().__init__()

        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")

        x_probe = np.asarray(x_probe, dtype=np.float32)
        if x_probe.ndim != 2 or len(x_probe) < 2:
            raise ValueError("Expected a 2D monitoring dataset")

        self.x_probe = x_probe
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.csv_path = csv_path

        self.records = []

        # Input statistics do not change during AE training.
        self.input_stats = spectrum_metrics(self.x_probe)

        # Fixed input PCA basis, used to find which
        # directions are reconstructed poorly.
        centered = self.x_probe - self.x_probe.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered.astype(np.float64), full_matrices=False,)
        self.pc_directions = vh[:8]

        projected = centered @ self.pc_directions.T
        self.pc_variances = np.mean(projected**2, axis=0)

    def on_train_begin(self, logs=None):
        # Shared layers and weights, without copying the network.
        self.probe_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer("bottleneck").output, self.model.outputs[0],],
        )

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        # Inference only: no training or weight updates.
        z_bottleneck, x_recon = self.probe_model.predict(self.x_probe, batch_size=self.batch_size, verbose=0)
        input_stats = self.input_stats
        bottleneck_stats = spectrum_metrics(z_bottleneck)
        recon_stats = spectrum_metrics(x_recon)
        reconstruction = reconstruction_metrics(self.x_probe, x_recon)

        metrics = {}

        # Representation spectra
        for prefix, stats in [
            ("input", input_stats),
            ("bottleneck", bottleneck_stats),
            ("reconstruction", recon_stats),
        ]:
            for key, value in stats.items():
                metrics[f"{prefix}_{key}"] = float(value)

        # Reconstruction quality
        for key, value in reconstruction.items():
            metrics[f"reconstruction_{key}"] = value

        # Representation similarity
        metrics["cka_input_bottleneck"] = linear_cka(self.x_probe, z_bottleneck)
        metrics["cka_input_reconstruction"] = linear_cka(self.x_probe, x_recon)

        # Reconstruction R2 for individual input PCs
        residual = x_recon - self.x_probe
        projected_residual = (residual @ self.pc_directions.T)

        for i, variance in enumerate(self.pc_variances):
            if variance < 1e-12:
                continue

            pc_mse = np.mean(projected_residual[:, i] ** 2)
            metrics[f"pc{i+1}_r2"] = float(1.0 - pc_mse / variance)

        # Save independent history
        self.records.append({"epoch": epoch + 1, **metrics,})

        # Make available to other Keras callbacks
        # and the returned History where supported.
        if logs is not None:
            logs.update(metrics)

    def on_train_end(self, logs=None):
        if self.csv_path is None or not self.records:
            return

        path = Path(self.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys(),)
            writer.writeheader()
            writer.writerows(self.records)


class EncoderMetricsCallback(keras.callbacks.Callback):
    """Monitor the representation produced by a model's encoder pool layer."""

    def __init__(
        self,
        x_probe: np.ndarray,
        layer_name: str = "pool",
        batch_size: int = 256,
        every_n_epochs: int = 1,
        csv_path: str | None = None,
        y_probe: np.ndarray | None = None,
        pca_rank: int | None = None,
    ):
        super().__init__()

        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if pca_rank is not None and pca_rank < 1:
            raise ValueError("pca_rank must be positive")

        x_probe = np.asarray(x_probe, dtype=np.float32)
        if x_probe.ndim < 2 or len(x_probe) < 2:
            raise ValueError("Expected at least two monitoring samples")

        self.x_probe = x_probe
        self.layer_name = layer_name
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.csv_path = csv_path
        self.pca_rank = pca_rank
        self.records = []
        self.previous_embedding = None
        self.previous_normalized_embedding = None

        if y_probe is None:
            self.class_labels = None
        else:
            y_probe = np.asarray(y_probe)
            if len(y_probe) != len(x_probe):
                raise ValueError("x_probe and y_probe must contain the same samples")
            if y_probe.ndim == 2:
                self.class_labels = np.argmax(y_probe, axis=1)
            elif y_probe.ndim == 1:
                self.class_labels = y_probe
            else:
                raise ValueError("Expected 1D labels or 2D one-hot labels")

    def on_train_begin(self, logs=None):
        pool_output = self.model.get_layer(self.layer_name).output
        if len(pool_output.shape) != 2:
            raise ValueError(
                f"Expected '{self.layer_name}' to produce a 2D embedding, "
                f"received shape {pool_output.shape}"
            )

        # This shares the trained model's layers and weights; it does not copy
        # or independently update the encoder.
        self.probe_model = keras.Model(
            inputs=self.model.inputs,
            outputs=pool_output,
            name="encoder_metrics_probe",
        )

    def _representation_metrics(
        self,
        embedding: np.ndarray,
        prefix: str,
    ) -> dict:
        metrics = {
            f"{prefix}_{key}": float(value)
            for key, value in spectrum_metrics(
                embedding,
                explained_variance_rank=self.pca_rank,
            ).items()
        }

        centered = embedding - embedding.mean(axis=0, keepdims=True)
        dimension_variances = np.var(centered, axis=0, ddof=1)
        sample_norms = np.linalg.norm(embedding, axis=1)

        metrics.update(
            {
                f"{prefix}_mean_dimension_variance": float(
                    np.mean(dimension_variances)
                ),
                f"{prefix}_min_dimension_variance": float(
                    np.min(dimension_variances)
                ),
                f"{prefix}_max_dimension_variance": float(
                    np.max(dimension_variances)
                ),
                f"{prefix}_inactive_dimension_fraction": float(
                    np.mean(dimension_variances < 1e-12)
                ),
                f"{prefix}_mean_l2_norm": float(np.mean(sample_norms)),
                f"{prefix}_std_l2_norm": float(np.std(sample_norms)),
            }
        )
        return metrics

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        embedding = self.probe_model.predict(
            self.x_probe,
            batch_size=self.batch_size,
            verbose=0,
        )
        embedding = np.asarray(embedding, dtype=np.float64)

        sample_norms = np.linalg.norm(embedding, axis=1)
        normalization_scale = np.sqrt(
            np.maximum(
                np.sum(embedding**2, axis=1, keepdims=True),
                1e-12,
            )
        )
        normalized_embedding = embedding / normalization_scale

        metrics = self._representation_metrics(embedding, "embedding")
        metrics.update(
            self._representation_metrics(
                normalized_embedding,
                "normalized_embedding",
            )
        )

        if self.class_labels is not None:
            metrics["embedding_fisher_ratio"] = fisher_separation_ratio(
                embedding,
                self.class_labels,
            )
            metrics[
                "normalized_embedding_fisher_ratio"
            ] = fisher_separation_ratio(
                normalized_embedding,
                self.class_labels,
            )

        if self.previous_embedding is None:
            metrics["embedding_epoch_to_epoch_cka"] = 1.0
            metrics["normalized_embedding_epoch_to_epoch_cka"] = 1.0
        else:
            metrics["embedding_epoch_to_epoch_cka"] = linear_cka(
                self.previous_embedding,
                embedding,
            )
            metrics["normalized_embedding_epoch_to_epoch_cka"] = linear_cka(
                self.previous_normalized_embedding,
                normalized_embedding,
            )

        self.previous_embedding = embedding
        self.previous_normalized_embedding = normalized_embedding
        self.records.append({"epoch": epoch + 1, **metrics})

        # Add the metrics to Keras History and make them available to any
        # callbacks that run after this one.
        if logs is not None:
            logs.update(metrics)

    def on_train_end(self, logs=None):
        if self.csv_path is None or not self.records:
            return

        path = Path(self.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
