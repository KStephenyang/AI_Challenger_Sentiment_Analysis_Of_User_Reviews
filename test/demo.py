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
import pandas as pd
import os

from sklearn.metrics import f1_score
import numpy as np
from app.models.algorithm.text_lstm import ChatTextLSTM
from app.models.clean import get_model_data, word2vec_train, word_cut, input_transform
from app.utils import setting
from app.utils.setting import PROJECT_PATH


def load_demo_data():
    demo_data_path = os.path.join(PROJECT_PATH, 'test/input/demo_data.csv')
    print(demo_data_path)
    demo_data = pd.read_csv(open(demo_data_path))
    return demo_data


def demo_lstm():
    demo_data = load_demo_data()
    print(demo_data.shape[0])
    demo_train = demo_data.iloc[:int(0.8 * demo_data.shape[0])]
    demo_valid = demo_data.iloc[int(0.8 * demo_data.shape[0]):]
    text_fullset = demo_data.iloc[:, 1]
    x_cut = word_cut(text_fullset)
    index_dict, word_vectors, x_combined = word2vec_train(x_cut)
    n_symbols, embedding_weights = get_model_data(index_dict, word_vectors)
    print(demo_train.shape, demo_train.shape)
    text_lstm = ChatTextLSTM(input_dim=n_symbols, embedding_dim=setting.VOCABULARY_VECTOR_DIM,
                             embedding_weights=embedding_weights)
    x_train = x_combined[:int(demo_train.shape[0])]
    x_valid = x_combined[int(demo_train.shape[0]):]
    f1_score_dict = dict()
    for col in demo_data.columns[2:]:
        y_train = demo_train[col] + 2
        y_valid = demo_valid[col] + 2
        text_lstm.train(x_train, y_train, x_valid, y_valid)
        print(y_valid)
        y_valid_pred = pd.Series([0] * x_valid.shape[0])
        for ind in range(x_valid.shape[0]):
            y_pred = np.argmax(text_lstm.model.predict(x_valid[ind].reshape(1, -1)))
            print(text_lstm.model.predict(x_valid[ind].reshape(1, -1)))
            print(y_pred)
            y_valid_pred[ind] = y_pred
        print(y_valid_pred)
        f1_score(y_valid, y_valid_pred, average='macro')
        f1_score_dict[col] = f1_score(y_valid, y_valid_pred, average='macro')
        print(f1_score_dict[col])
        text_lstm.model.save('lstm_model_demo_{}.h5'.format(col))
    f1_score_mn = np.mean(list(f1_score_dict.values()))
    print(f1_score_mn)
    return True


if __name__ == '__main__':
    print(demo_lstm())
