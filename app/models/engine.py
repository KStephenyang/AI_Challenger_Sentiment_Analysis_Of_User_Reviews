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
from gensim.models import Word2Vec
from keras.models import load_model
from app.services.chatter.ibns.app.models.algorithm import IntentClassifier, DomainClassifier
from app.services.chatter.ibns.app.models.reply.replier import ReplierBase
from app.services.chatter.ibns.app.models.setting import VAR_DOMAIN, ONLINE_MODELS_DIR, VAR_INTENT, WORD_VECTOR_CORPUS
from app.services.chatter.ibns.app.models.setting import DOMAIN_TRAIN_COLS, INTENT_TRAIN_COLS, DOMAIN2NUM_MAP
from app.services.chatter.ibns.app.models.clean import get_corpus4train, id2num_dict
from app.services.chatter.ibns.app.models.setting import VERSION


def intent_trainer(X, domain):
    intent_list = sorted(set(X[VAR_INTENT].tolist()))
    intent2num_map = os.path.join(ONLINE_MODELS_DIR, 'chat_intent2num_{}_{}.csv'.format(domain, VERSION))
    with open(intent2num_map, 'w') as f:
        for ind, intent in enumerate(intent_list):
            f.writelines('{}:{}\n'.format(ind, intent))
    intent_dict = {intent: ind for ind, intent in enumerate(intent_list)}
    X[VAR_INTENT] = X[VAR_INTENT].map(intent_dict)
    print(X)
    intent_model = IntentClassifier(domain=domain)
    intent_model.fit(X=X)
    file_path = os.path.join(ONLINE_MODELS_DIR, 'intent_model_{}_{}.h5'.format(domain, VERSION))
    intent_model.dump(file_path=file_path)


def domain_trainer(X):
    domain_list = sorted(set(X[VAR_DOMAIN].tolist()))
    with open(DOMAIN2NUM_MAP, 'w') as f:
        for ind, domain in enumerate(domain_list):
            f.writelines('{}:{}\n'.format(ind, domain))
    domain_dict = {domain: ind for ind, domain in enumerate(domain_list)}
    X[VAR_DOMAIN] = X[VAR_DOMAIN].map(domain_dict)
    print(X)
    domain_model = DomainClassifier()
    domain_model.fit(X=X)
    file_path = os.path.join(ONLINE_MODELS_DIR, 'domain_model_{}.h5'.format(VERSION))
    domain_model.dump(file_path=file_path)


def trainer():
    corpus = get_corpus4train()
    domain_trainer(corpus[DOMAIN_TRAIN_COLS])
    for domain, task_intent in corpus.groupby(VAR_DOMAIN):
        intent_trainer(task_intent[INTENT_TRAIN_COLS], domain)
    print('success')


def chat_model_load():
    model_dict = dict()
    domain_path = os.path.join(ONLINE_MODELS_DIR, 'domain_model_{}.h5'.format(VERSION))
    model_dict[VAR_DOMAIN] = load_model(domain_path) if os.path.exists(domain_path) else None
    model_dict[VAR_INTENT] = dict()
    id2num_path = os.path.join(ONLINE_MODELS_DIR, 'chat_domain2num_{}.csv'.format(VERSION))
    domain_dict = id2num_dict(id2num_path) if os.path.exists(id2num_path) else None
    if domain_dict:
        for domain in domain_dict.values():
            intent_path = os.path.join(ONLINE_MODELS_DIR, 'intent_model_{}_{}.h5'.format(domain, VERSION))
            model_dict[VAR_INTENT][domain] = load_model(intent_path) if os.path.exists(intent_path) else None
    word2vec_model = Word2Vec.load(WORD_VECTOR_CORPUS) if os.path.exists(WORD_VECTOR_CORPUS) else None
    return model_dict, word2vec_model


class OnlineReplier:
    model_dict, word2vec = chat_model_load()

    def __init__(self):
        pass

    def reply(self, question_):
        replier = ReplierBase(question_, self.model_dict, self.word2vec)
        return replier.reply()


if __name__ == '__main__':
    # trainer()
    replier_ = OnlineReplier()
    while 1:
        question = input('question:')
        print('answer:', replier_.reply(question))
