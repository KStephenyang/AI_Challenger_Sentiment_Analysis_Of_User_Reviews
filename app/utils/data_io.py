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
from configparser import ConfigParser
from app.utils.setting import CONFIG_PATH
import pandas as pd


def config_parser():
    cf = ConfigParser()
    cf.read(CONFIG_PATH, encoding='utf-8')
    return cf


def load_train_set():
    parser = config_parser()
    train_set_path = parser.get('DataPara', 'TRAIN_SET_PATH')
    train_set = pd.read_csv(open(train_set_path, encoding='utf-8'))
    return train_set


def load_valid_set():
    parser = config_parser()
    valid_set_path = parser.get('DataPara', 'VALID_SET_PATH')
    valid_set = pd.read_csv(open(valid_set_path, encoding='utf-8'))
    return valid_set


def load_test_set():
    parser = config_parser()
    test_set_path = parser.get('DataPara', 'TEST_SET_PATH')
    test_set = pd.read_csv(open(test_set_path, encoding='utf-8'))
    return test_set


if __name__ == '__main__':
    load_train_set()
    load_valid_set()
    load_test_set()
