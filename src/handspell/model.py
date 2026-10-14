# Imports for general model training
import numpy as np
import pandas as pd
import keras

from tensorflow.keras import Sequential, Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from handspell.data_preprocessing import image_preprocessing


def dataset_load(train_path: str, test_path: str):
    """Function to load both datasets"""

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    return train, test

def target_separation(df, target_label):
    """Function to separate target and features"""

    df = df.copy()

    X = df.drop(columns=[target_label])
    y = df[[target_label]]

    return X, y


def feature_normalizing(X):
    """Function to normalize, reshape features (Original dataset is flatten 784, 28x28)"""

    n_samples = X.shape[0]

    X_norm = X.values.reshape(n_samples, 28, 28, 1) / 255.0

    return X_norm

def validation_split(X, y, test_size: float=0.2, random_state: int=1):
    """Function to apply a train_test_split on test data to create a validation set from it"""

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    return X_train, X_test, y_train, y_test


def dataset_preprocessing(train, test):
    """Function to call both training and test datasets and assign variables"""

    target_label = 'label'

    X_train, y_train = target_separation(train, target_label)
    X_test, y_test = target_separation(test, target_label)

    X_train = feature_normalizing(X_train)
    y_train = y_train.values
    X_test = feature_normalizing(X_test)
    y_test = y_test.values

    return X_train, y_train, X_test, y_test


def init_model(X_train):
    """Function to initiliaze and compile model, default values set to original baseline model"""

    model = Sequential()

    model.add(Input(shape=X_train.shape[1:]))

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(64, (3, 3), padding='same' ,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ### Third Convolution & MaxPooling
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(256, activation='relu'))

    ### Dropout layer - rate = 0.4, fraction of units to drop
    model.add(layers.Dropout(0.4))

    ### Last layer - Classification Layer with 25 outputs corresponding to 25 digits
    model.add(layers.Dense(25, activation='softmax'))

    ### Model compilation
    model.compile(
            loss='sparse_categorical_crossentropy', # sparse if not to_categorical
            optimizer='adam',
            metrics=['accuracy'])

    return model


def train_model(X_train, y_train, X_val, y_val):
    """Function to set Early Stopping/ReduceLR and train model"""

    es = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.4,
                                  patience=3,
                                  verbose=1,
                                  min_lr=0.00001)

    model = init_model(X_train)

    history = model.fit(X_train, y_train,
              batch_size=64,
              epochs=50,
              callbacks=[reduce_lr],
              validation_data=(X_val, y_val),
              verbose=1)

    return model, history


def evaluate_model(X_test, y_test, model):
    """Function to evaluate model"""

    return model.evaluate(X_test, y_test, verbose=1)


def save_model(model, model_file_name: str="models/model_hand28x28_letter.keras"):
    """Function to save model"""

    model.save(model_file_name)

    return

def load_model(path: str):
    """Function to load model"""

    loaded_model = keras.models.load_model(path)

    return loaded_model


def predict_letter(scores: np.ndarray) -> dict:

    predicted_idx = int(np.argmax(scores))

    return {
        "letter": chr(predicted_idx + ord("A")),
        "confidence": float(scores[predicted_idx])
    }


def predict(image_path: str, model_path: str):
    """"""
    preprocessed_img = image_preprocessing(image_path, 28)

    normalized_img = preprocessed_img.astype(np.float32) / 255.0

    reshape_img = normalized_img.reshape(1, 28, 28, 1)

    model = load_model(model_path)

    predictions = model.predict(reshape_img)

    predicted_letter = predict_letter(predictions[0])

    return predicted_letter, predictions
