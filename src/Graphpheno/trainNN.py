from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
import numpy as np
from sklearn import svm
import time


def train_nn(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(1024, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    model.add(Dense(Y_train.shape[1],activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=0)

    y_prob = model.predict(X_test)

    model.save('my_model.h5')

    # model = load_model('my_model.h5')

    return y_prob


def train_svm(X_train, Y_train, X_test, Y_test):
    time1 = time.time()
    clf = svm.SVC()
    y_prob = np.ones((X_train.shape[1], Y_train.shape[1]))
    for i in range(Y_train.shape[1]):
        if 0 < sum(Y_train[:, i]) < Y_train.shape[0]:
           clf.fit(X_train, Y_train[:, i])
           y_prob[:, i] = clf.predict(X_test)

    return y_prob






    


