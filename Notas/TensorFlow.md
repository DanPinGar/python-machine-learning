
- TensorFlow python API documentation: https://www.tensorflow.org/api_docs/python/tf/all_symbols

- Keras Guide: https://www.tensorflow.org/guide/keras

- KerasCV: https://keras.io/keras_cv/
# Codes

### TensorFlow tutorials:


1 - Convolutional Neural Network (CNN): https://www.tensorflow.org/tutorials/images/cnn

```
model = models.Sequential()model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))model.add(layers.MaxPooling2D((2, 2)))model.add(layers.Conv2D(64, (3, 3), activation='relu'))model.add(layers.MaxPooling2D((2, 2)))model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())model.add(layers.Dense(64, activation='relu'))model.add(layers.Dense(10))
```

2- Image classification: https://www.tensorflow.org/tutorials/images/classification


3 - Data augmwntation: https://www.tensorflow.org/tutorials/images/data_augmentation

EDA & Keras CNN Tutorial | Cats vs Dogs:

https://www.kaggle.com/code/srispirit/eda-keras-cnn-tutorial-cats-vs-dogs



