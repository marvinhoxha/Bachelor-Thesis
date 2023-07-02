import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import regularizers
from PIL import ImageFile
import pickle
import os
import logging
import pickle
import mlflow
from tensorflow.keras.callbacks import EarlyStopping
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
LR = 6e-4
BATCH_SIZE = 32
EPOCHS = 150
IMG_SIZE = 244  # Updated image size

def get_train_generator():
    """Get the Train Path"""
    data_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
    )
    return data_datagen.flow_from_directory(
        "improved_dataset/train/",
        target_size=(IMG_SIZE, IMG_SIZE),  # Updated target size
        batch_size=BATCH_SIZE,
    )


def get_valid_generator():
    """Get the Valid Path"""
    data_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return data_datagen.flow_from_directory(
        "improved_dataset/valid/",
        target_size=(IMG_SIZE, IMG_SIZE),  # Updated target size
        batch_size=BATCH_SIZE,
    )


def train():
    """Train the model"""
    logging.info("Training Model.")

    resnet_body = tf.keras.applications.ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),  # Updated input shape
    )
    resnet_body.trainable = False
    unfreeze_layers = 15  # Number of blocks to unfreeze
    for layer in resnet_body.layers[-unfreeze_layers:]:
        layer.trainable = True
    # early_stopping = EarlyStopping(
    # monitor='val_loss',  # Metric to monitor for early stopping
    # patience=5,          # Number of epochs with no improvement before stopping
    # restore_best_weights=True  # Restores the best weights based on the monitored metric
    # )   

    # inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))  # Updated input shape
    # x = resnet_body(inputs, training=False)
    # x = tf.keras.layers.Flatten()(x)
    # outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    # resnet_model = tf.keras.Model(inputs, outputs)
    resnet_model = tf.keras.Sequential([
        resnet_body,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    resnet_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LR),
        loss=tf.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    train_generator = get_train_generator()
    valid_generator = get_valid_generator()

    logging.info(resnet_body.summary())
    logging.info("\n\n")
    logging.info(resnet_model.summary())
    mlflow.autolog()

    resnet_model.fit(
        train_generator, epochs=EPOCHS, validation_data=valid_generator
    )

    labels = train_generator.class_indices

    logging.info("Dump models.")
    resnet_model.save("./models/XrayModel_resnet/1")

    with open("./models/labels.pickle", "wb") as handle:
        pickle.dump(labels, handle)
    logging.info("Finished training.")


if __name__ == "__main__":
    train()