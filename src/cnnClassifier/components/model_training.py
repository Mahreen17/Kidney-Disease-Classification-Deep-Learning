import tensorflow as tf
from pathlib import Path

from cnnClassifier.entity.config_entity import TrainingConfig


class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):

        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        # VGG16 ImageNet preprocessing
        datagenerator_kwargs = dict(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # ==============================
        # VALIDATION DATA GENERATOR
        # ==============================

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            class_mode="categorical",
            **dataflow_kwargs
        )

        # ==============================
        # TRAINING DATA GENERATOR
        # ==============================

        if self.config.params_is_augmentation:

            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
                validation_split=0.20,

                rotation_range=20,
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1
            )

        else:

            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            class_mode="categorical",
            **dataflow_kwargs
        )

        # ==============================
        # DISPLAY DATASET INFORMATION
        # ==============================

        print("Class indices:", self.train_generator.class_indices)
        print("Training images:", self.train_generator.samples)
        print("Validation images:", self.valid_generator.samples)

    def train(self):

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):

        model.save(path)