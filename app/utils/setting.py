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
import os
from configparser import ConfigParser

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROJECT_NAME = os.path.basename(PROJECT_PATH)
CONFIG_PATH = os.path.join(PROJECT_PATH, 'cfg/config.ini')
cf = ConfigParser()
cf.read(CONFIG_PATH)
VERSION = cf.get('ReleaseNotes', 'VERSION')

MODELS_SAVE_DIR = os.path.join(u'{}'.format(PROJECT_PATH), 'result/models_{}'.format(VERSION))
if not os.path.exists(MODELS_SAVE_DIR):
    os.makedirs(MODELS_SAVE_DIR)

VOCABULARY_SIZE = 200
VOCABULARY_MAXLEN = 20
VOCABULARY_CORPUS = os.path.join(MODELS_SAVE_DIR, 'chat_vocabulary_{}.csv'.format(VERSION))
WORD_VECTOR_CORPUS = os.path.join(MODELS_SAVE_DIR, 'chat_word2vec_model_{}.pkl'.format(VERSION))
WORD_DICT_CORPUS = os.path.join(MODELS_SAVE_DIR, 'chat_word2vec_dict_{}.csv'.format(VERSION))
DOMAIN2NUM_MAP = os.path.join(MODELS_SAVE_DIR, 'chat_domain2num_{}.csv'.format(VERSION))
VOCABULARY_VECTOR_DIM = 300
BATCH_SIZE = 32
N_EPOCH = 1
MIN_WORD_COUNTS = 1
WINDOW_SIZE = 5
CPU_COUNTS = 4
N_ITERATIONS = 1

ALGORITHM = 'LSTM'  # LSTM, CNN, DNN
