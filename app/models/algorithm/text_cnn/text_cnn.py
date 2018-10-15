# ------------------------------------------------------------
#  产品：****
#  版本：0.1
#  版权：****
#  模块：****
#  功能：****
#  语言：Python3.6
#  作者：****<****@aconbot.com.cn>
#  日期：2018-08-25
# ------------------------------------------------------------
#  修改人：****<****@aconbot.com.cn>
#  修改日期：2018-08-26
#  修改内容：创建
# ------------------------------------------------------------
from gensim.models import Word2Vec
from keras import Sequential, Model, Input
from keras.layers import Embedding, Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Convolution1D, MaxPool1D, concatenate
from keras.models import load_model
from keras.preprocessing import sequence
from keras.utils import np_utils
import numpy as np
from app.models.clean import word_cut, input_transform, word2vec_train
from app.utils import setting
from app.models.clean import get_model_data
from app.utils.setting import WORD_VECTOR_CORPUS
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ChatCNN:
    def __init__(self, input_dim, embedding_dim=None, embedding_weights=None):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding_weights = embedding_weights
        self.model = None

    def cnn_create(self, n_classes, input_shape):
        self.model = Sequential()
        if self.embedding_weights is not None:
            self.model.add(Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim, input_shape=input_shape,
                                     weights=[self.embedding_weights]))  # Adding Input Length
        else:
            self.model.add(Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim,
                                     input_shape=input_shape))  # Adding Input Length
        self.model.add(Conv1D(256, 3, padding='same', activation='relu', strides=1))
        self.model.add(MaxPooling1D(3, 3, padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(self.embedding_dim, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='softmax'))
        print('Compiling the Model...')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        return self

    def train(self, x_train, y_train, x_test, y_test):
        print('Defining a Simple Keras Model...')
        n_classes = len(set(y_train.tolist() + y_test.tolist()))
        print(n_classes)
        y_train = np_utils.to_categorical(y_train, n_classes)
        y_test = np_utils.to_categorical(y_test, n_classes)
        print((x_train.shape[1],))
        self.cnn_create(n_classes, (x_train.shape[1],))
        print("Train...")
        self.model.fit(x_train, y_train, batch_size=setting.BATCH_SIZE, epochs=setting.N_EPOCH, verbose=1,
                       validation_data=(x_test, y_test))
        print("Evaluate...")
        score = self.model.evaluate(x_test, y_test, batch_size=setting.BATCH_SIZE)
        print('Test score:', score)
        return self

    def predict(self):
        pass


class ChatTextCNN:
    def __init__(self, input_dim, embedding_dim=None, embedding_weights=None):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding_weights = embedding_weights
        self.model = None

    def cnn_create(self, n_classes, input_shape):
        # self.model = Sequential()
        main_input = Input(shape=input_shape)
        if self.embedding_weights is not None:
            embed = Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim,
                              weights=[self.embedding_weights])  # Adding Input Length
        else:
            embed = Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim)  # Adding Input Length
        embed = embed(main_input)
        # 词窗大小分别为3,4,5
        cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = MaxPool1D(pool_size=4)(cnn1)
        cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = MaxPool1D(pool_size=4)(cnn2)
        cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = MaxPool1D(pool_size=4)(cnn3)
        # 合并三个模型的输出向量
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = Flatten()(cnn)
        drop = Dropout(0.2)(flat)
        main_output = Dense(n_classes, activation='softmax')(drop)
        self.model = Model(inputs=main_input, outputs=main_output)
        print('Compiling the Model...')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        return self

    def train(self, x_train, y_train, x_test, y_test):
        print('Defining a Simple Keras Model...')
        n_classes = len(set(y_train.tolist() + y_test.tolist()))
        print(n_classes)
        y_train = np_utils.to_categorical(y_train, n_classes)
        y_test = np_utils.to_categorical(y_test, n_classes)
        print((x_train.shape[1],))
        self.cnn_create(n_classes, (x_train.shape[1],))
        print("Train...")
        self.model.fit(x_train, y_train, batch_size=setting.BATCH_SIZE, epochs=setting.N_EPOCH, verbose=1,
                       validation_data=(x_test, y_test))
        print("Evaluate...")
        score = self.model.evaluate(x_test, y_test, batch_size=setting.BATCH_SIZE)
        print('Test score:', score)
        return self

    def predict(self):
        pass


def cnn_train(X):
    x_cut = word_cut(X)
    index_dict, word_vectors, x_combined = word2vec_train(x_cut)
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_model_data(index_dict,
                                                                                    word_vectors,
                                                                                    x_combined,
                                                                                    X.iloc[:, 1])
    text_cnn = ChatTextCNN(input_dim=n_symbols, embedding_dim=setting.VOCABULARY_VECTOR_DIM,
                           embedding_weights=embedding_weights)
    text_cnn.train(x_train, y_train, x_test, y_test)
    return text_cnn.model


def cnn_predict(string_, model_path='cnn_model_test_0.h5'):
    print('loading models_v1.0......')
    model = load_model(model_path)
    word2vec_model = Word2Vec.load(WORD_VECTOR_CORPUS)
    data = input_transform(string_, word2vec_model)
    data.reshape(1, -1)
    data = sequence.pad_sequences(data, maxlen=setting.VOCABULARY_MAXLEN)
    print(data)
    result = model.predict(data)
    print(result)
    print(np.argmax(result[0]))
    return np.argmax(result[0])


if __name__ == '__main__':
    pass
