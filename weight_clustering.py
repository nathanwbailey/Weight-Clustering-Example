import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import numpy as np
import os
import zipfile

def get_compressed_model_size(file, zipped_file):
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential(
    [
        keras.layers.Reshape(target_shape=(28,28,1), input_shape=(28,28)),
        keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ]
)

model.compile(
    optimizer='sgd', 
    loss="sparse_categorical_crossentropy", 
    metrics=['accuracy']
)

model.summary()

model.fit(
    train_images,
    train_labels,
    validation_split=0.2,
    epochs=10
)

test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels
)

print(f'Baseline Test Loss: {test_loss}')
print(f'Baseline Test Accuracy: {test_accuracy}')

keras.models.save_model(model, 'baseline_model.h5', include_optimizer=False)

clustered_model = tfmot.clustering.keras.cluster_weights(
    model,
    number_of_clusters = 16 ,
    cluster_centroids_init = tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS
)

clustered_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1e-5), 
    loss="sparse_categorical_crossentropy", 
    metrics=['accuracy']
)

clustered_model.summary()

clustered_model.fit(
    train_images,
    train_labels,
    validation_split=0.2,
    epochs=10
)

clustered_model_test_loss, clustered_model_test_accuracy = model.evaluate(
    test_images,
    test_labels
)

print(f'Clustered Model Test Loss: {clustered_model_test_loss}')
print(f'Clustered Model Test Accuracy: {clustered_model_test_accuracy}')

final_model = tfmot.clustering.keras.strip_clustering(clustered_model)

for layer in final_model.layers:
    for weight in layer.weights:
        if 'kernel:0' in weight.name:
            print((f"Number of clusters in weight {layer.name}/{weight.name} is {len(np.unique(weight))}"))

keras.models.save_model(final_model, 'clustered_model.h5', include_optimizer=False)

final_model_tflite = tf.lite.TFLiteConverter.from_keras_model(final_model).convert()

with open('clustered_model.tflite', 'wb') as f:
    f.write(final_model_tflite)

print(f"Size of zipped baseline keras model: {get_compressed_model_size('baseline_model.h5', 'baseline_model_keras_zipped'):.2f}")

print(f"Size of zipped clustered keras model: {get_compressed_model_size('clustered_model.h5', 'clustered_model_keras_zipped'):.2f}")

print(f"Size of zipped clustered tflite model: {get_compressed_model_size('clustered_model.tflite', 'clustered_model_tflite_zipped'):.2f}")
