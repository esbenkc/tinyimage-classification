from keras.models import load_model
import sys
import os
import imageio
from keras.callbacks import ReduceLROnPlateau  # learning rate decay policy
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
# Import the Model class
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.client import device_lib
import datetime
from zipfile import ZipFile
from pathlib import Path
import time
import pandas as pd  # for making our csv
import numpy as np
from sklearn.model_selection import train_test_split  # for splitting data
# if label is 0,1,...,99 etc then it becomes [0,...1,.,0] a len 100 vector
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential  # This building the models
from keras.preprocessing.image import ImageDataGenerator  # Data Augmentation
from keras.regularizers import l2
from tensorflow import keras
from keras import backend as k
import tensorflow as tf
import random as rn
import seaborn as sns
from matplotlib import style
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)


def plot_accuracy_loss(history):
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def classification_report(model, x_test, y_test, n_classes=100):
    # get the predictions for the test data
    predicted_classes = model.predict_classes(x_test)

    # get the indices to be plotted
    correct = np.nonzero(predicted_classes == y_test)[0]
    incorrect = np.nonzero(predicted_classes != y_test)[0]
    target_names = ["Class {}".format(i) for i in range(n_classes)]
    print(classification_report(
        y_test, predicted_classes, target_names=target_names))

    # plot some of the misclassified examples
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_test[correct].reshape(64, 64),
                   cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(
            predicted_classes[correct], y_true[correct]))
        plt.tight_layout()

    for i, incorrect in enumerate(incorrect[0:9]):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_test[incorrect].reshape(64, 64),
                   cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(
            predicted_classes[incorrect], y_true[incorrect]))
        plt.tight_layout()

def get_id_dictionary():
    """
    Maps each class id to an unique integer.
    """
    id_dict = {}
    for i, line in enumerate(open(path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict


def get_class_to_id_dict():
    """
    Maps each class id to the English version of the label.
    """
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open(path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])
    return result


def get_data(id_dict, n_samples=10):
    """
    n_samples: number of samples per class. n_samples has a max of 500.
    The range is [1, 500].
    """
    print('starting loading data')
    train_data, test_data = [], []
    train_labels = []
    t = time.time()
    for key, value in id_dict.items():
        if value < 100:  # Only focus on first 100 classes
            train_data += [imageio.imread(path + 'train/train/{}/images/{}_{}.JPEG'.format(
                key, key, str(i)), pilmode='RGB') for i in range(n_samples)]
            train_labels_ = np.array([[0]*100]*n_samples)
            train_labels_[:, value] = 1
            train_labels += train_labels_.tolist()

    test_image_names = []
    path_list = list(Path(path+'test/test/images/').glob('*.jpg'))
    for test_image_path in path_list:
        img_name = str(test_image_path).split('.')[0][-18:]
        test_image_names.append(img_name)
        test_data.append(imageio.imread(test_image_path, pilmode='RGB'))

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return np.array(train_data), np.array(train_labels), np.array(test_data), test_image_names


def create_submission_file(model, x_test_names, name="submission.csv"):
    name2idx = {}
    sample_submission = pd.read_csv(path + 'submission_sample.csv')
    filename_order = sample_submission['img_id'].values
    for i in range(len(filename_order)):
        name2idx[filename_order[i]] = i

    # Google colab reads the files in a different order than the answer file was created.
    # This is done to preserve the file order.
    result_dict = {'img_id': [None]*len(x_test),
                   'label': [None]*len(x_test)}
    test_preds = np.argmax(model.predict(x_test/255.), axis=-1)

    for i in range(len(x_test_names)):
        test_image_name = x_test_names[i]
        result_dict['img_id'][name2idx[test_image_name]] = test_image_name
        result_dict['label'][name2idx[test_image_name]] = test_preds[i]

    pd.DataFrame(result_dict).to_csv(name, index=False)


def create_deep_cnn(input_shape=(64, 64, 3), num_classes=100, regularize=False):
    cnn4 = Sequential()
    cnn4.add(Conv2D(32, kernel_size=(3, 3),
             activation='relu', input_shape=input_shape,
             kernel_regularizer=regularizers.l2(0.01) if regularize else None))
    cnn4.add(BatchNormalization())

    cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
             kernel_regularizer=regularizers.l2(0.01) if regularize else None))
    cnn4.add(BatchNormalization())
    cnn4.add(MaxPooling2D(pool_size=(2, 2)))
    cnn4.add(Dropout(0.25))

    cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
             kernel_regularizer=regularizers.l2(0.01) if regularize else None))
    cnn4.add(BatchNormalization())
    cnn4.add(Dropout(0.25))

    cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
             kernel_regularizer=regularizers.l2(0.01) if regularize else None))
    cnn4.add(BatchNormalization())
    cnn4.add(MaxPooling2D(pool_size=(2, 2)))
    cnn4.add(Dropout(0.25))

    cnn4.add(Flatten())

    cnn4.add(Dense(512, activation='relu',
             kernel_regularizer=regularizers.l2(0.01) if regularize else None))
    cnn4.add(BatchNormalization())
    cnn4.add(Dropout(0.5))

    cnn4.add(Dense(128, activation='relu',
             kernel_regularizer=regularizers.l2(0.01) if regularize else None))
    cnn4.add(BatchNormalization())
    cnn4.add(Dropout(0.5))

    cnn4.add(Dense(num_classes, activation='softmax'))

    cnn4.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(lr=0.01),
                 metrics=['acc'])

    return cnn4


def create_shallow_cnn(input_shape=(64, 64, 3), num_classes=100):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.l1(0.01),
                     kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros(),
                     input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.l1(0.01),
                     kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the output of the convolutional layers into a 1-D vector for the fully connected layer.
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01),
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    bias_initializer=tf.keras.initializers.Zeros()))
    model.add(Dense(100, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.01),
              kernel_initializer=tf.keras.initializers.GlorotUniform(),
              bias_initializer=tf.keras.initializers.Zeros()))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(
                      learning_rate=0.01, momentum=0.1, nesterov=False, name="SGD"),
                  metrics=['acc'])

    return model


def set_data_generators(x_train, y_train, x_test, y_test, batch_size=20):
    # Set up data generators for training and validation set
    datagen = ImageDataGenerator(
        rescale=1/255.,
        featurewise_center=False,           # set input mean to 0 over the dataset
        samplewise_center=False,            # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,                # apply ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=8,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.3,
        zoom_range=0.1,
        horizontal_flip=True,               # randomly flip images
        vertical_flip=True)               # randomly flip images

    valid_datagen = ImageDataGenerator(rescale=1/255.)

    # fit generators to datasets
    train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    valid_generator = valid_datagen.flow(x_val, y_val, batch_size=batch_size)

    return train_generator, valid_generator


def finetune_MobileNetV2(x_train, y_train, x_val, y_val, x_test_names, batch_size=20, input_shape=(64, 64, 3), classes=100, transfer_layer_name='block5_pool', freeze=True, epochs_per_step=5):
    base_model = MobileNetV2(input_shape=input_shape,  # Shape of our images
                             include_top=False,  # Leave out the last fully connected layer
                             weights='imagenet',
                             classes=classes)

    # Freeze the layers except the last 4 layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a convolutional layer
    x = base_model.output
    x = Conv2D(filters=512, kernel_size=(3, 3),
               activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Check the trainable status of the individual layers
    for layer in base_model.layers:
        print(layer, layer.trainable)

    # Add the last fully connected layer
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(classes, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # model = load_model('cnn_mobilenet_model.h5')
    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  metrics=['acc', 'categorical_accuracy'])

    train_generator, valid_generator = set_data_generators(
        x_train, y_train, x_val, y_val, batch_size=batch_size)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=len(x_train)//batch_size,
                                  epochs=epochs_per_step,
                                  validation_data=valid_generator,
                                  validation_steps=len(x_val)//batch_size,
                                  callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='cnn_mobilenet_model.h5',
                                                                                monitor='val_loss', save_best_only=True),
                                             tensorboard_callback
                                             ])

    # Unfreeze all layers
    for layer in model.layers[:-4]:
        layer.trainable = True

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=len(x_train)//batch_size,
                                  epochs=epochs_per_step,
                                  validation_data=valid_generator,
                                  validation_steps=len(x_val)//batch_size,
                                  callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='cnn_mobilenet_model.h5',
                                                                                monitor='val_loss', save_best_only=True),
                                             tensorboard_callback
                                             ])

    create_submission_file(model, x_test_names)
    return history, model


if __name__ == '__main__':
    path = './'
    x_train, y_train, x_test, test_image_names = get_data(
        get_id_dictionary(), n_samples=500)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)
    print("train data shape: ",  x_train.shape)
    print("train label shape: ", y_train.shape)
    print("val data shape: ",  x_val.shape)
    print("val label shape: ", y_val.shape)
    print("test data shape: ",   x_test.shape)

    history, model = finetune_MobileNetV2(
        x_train, y_train, x_val, y_val, test_image_names)

