import tensorflow as tf
from pathlib import Path

from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:

    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):

        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(
            path=self.config.base_model_path,
            model=self.model
        )

    @staticmethod
    def _prepare_full_model(
        model,
        classes,
        freeze_all,
        freeze_till,
        learning_rate
    ):

        # First freeze all VGG16 layers
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False

        # Unfreeze VGG16 Block 4 and Block 5
        for layer in model.layers:
            if layer.name.startswith("block4") or layer.name.startswith("block5"):
                layer.trainable = True

        # Global Average Pooling
        global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()(
            model.output
        )

        # Final classification layer
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(global_average_pooling)

        # Create complete model
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile model
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()

        return full_model

    def update_base_model(self):

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )

    @staticmethod
    def save_model(
        path: Path,
        model: tf.keras.Model
    ):

        model.save(path)
        
        