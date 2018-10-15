# ------------------------------------------------------------
#  产品：****
#  版本：0.1
#  版权：****
#  模块：****
#  功能：****
#  语言：Python3.6
#  作者：****<****@aconbot.com.cn>
#  日期：2018-07-18
# ------------------------------------------------------------
#  修改人：****<****@aconbot.com.cn>
#  修改日期：2018-07-18
#  修改内容：创建
# ------------------------------------------------------------
from keras.models import load_model
from keras.preprocessing import sequence

from app.models.algorithm.text_dnn import dnn_train
from .text_cnn import cnn_train
from app.utils.setting import ALGORITHM, VOCABULARY_MAXLEN
from .text_lstm import lstm_train
import numpy as np


def domain_train(X):
    if ALGORITHM is 'CNN':
        return cnn_train(X)
    elif ALGORITHM is 'LSTM':
        return lstm_train(X)
    elif ALGORITHM is 'DNN':
        return dnn_train(X)
    else:
        return cnn_train(X)


class DomainClassifier:
    """
        识别提问所属Domain,为后续Intent识别提高准确率
    """

    def __init__(self):
        self.model = None

    def dump(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = load_model(file_path)

    def fit(self, X):
        self.model = domain_train(X)

    def predict(self, X):
        X = sequence.pad_sequences(X, maxlen=VOCABULARY_MAXLEN)
        result = self.model.predict(X)
        return np.argmax(result[0])

    def fit_predict(self, X):
        self.fit(X)
        self.predict(X)

    def write(self):
        pass
