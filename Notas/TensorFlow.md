
TensorFlow python API documentation: https://www.tensorflow.org/api_docs/python/tf/all_symbols


# Codes

1- Guia inicial de TensorFlow 2.0 para principiantes: https://www.tensorflow.org/tutorials/quickstart/beginner?hl=es-419

Modelo para distinguir números hechos a mano entre 0 y 9
```
model = tf.keras.models.Sequential([  
tf.keras.layers.Flatten(input_shape=(28, 28)),  
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.2),  
tf.keras.layers.Dense(10, activation='softmax')])
```

2- Basic classification: Classify images of clothing: https://www.tensorflow.org/tutorials/keras/classification

```
model = tf.keras.Sequential([    
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10)])
```




EDA & Keras CNN Tutorial | Cats vs Dogs:

https://www.kaggle.com/code/srispirit/eda-keras-cnn-tutorial-cats-vs-dogs



