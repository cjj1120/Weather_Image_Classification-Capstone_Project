import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet  import EfficientNetB4 

train_datagen= tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    shear_range=0.15,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.3,
    zoom_range=0.2,
    preprocessing_function=None
)

### LOAD DATA 
train_ds = train_datagen.flow_from_directory(
    r'C:\Users\User\@Code-ML\Zoom-camp Capstone Project\Data\dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training') # set as training data

val_ds = train_datagen.flow_from_directory(
    r'C:\Users\User\@Code-ML\Zoom-camp Capstone Project\Data\dataset', # same directory as training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation') # set as validation data

# Final model  
def make_model() :
    base_model= EfficientNetB4(weights='imagenet', include_top= False, input_shape=(224, 224, 3))
    base_model.trainable = False
    ##################################################
    inputs = keras.Input(shape=(224, 224, 3))   
    base = base_model(inputs, training= False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    drop1 = keras.layers.Dropout(0.5)(vectors)
    inner = keras.layers.Dense(100, activation= 'relu')(drop1) 
    drop2 = keras.layers.Dropout(0.5)(inner)
    outputs = keras.layers.Dense(8, activation='softmax')(drop2) 
    model = keras.Model(inputs, outputs)
    ##################################################
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(
        optimizer=optimizer, 
        loss = loss, 
        metrics=['accuracy']
    ) 
    return model 

chechpoint = keras.callbacks.ModelCheckpoint(
    'EfficientNetB4_Epoch-{epoch:02d}_Val-acc-{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

model = make_model()

history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[chechpoint]
)