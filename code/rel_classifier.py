import sys
import os
sys.path.append(os.getcwd())

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support

import config as cfg
from corpus.WLPDataset import WLPDataset


from collections import OrderedDict
import argparse



def single_run(x_train, y_train, x_test, y_test):
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=8, max_iter=1000)

    model.fit(x_train, y_train)
    pickle.dump(model, open('lr_model.m', 'wb'))

    print("Results on test set: ")
    pred_test = model.predict(x_test)

    print(classification_report(y_test, pred_test, target_names=cfg.RELATIONS, labels=range(len(cfg.RELATIONS))))
    print("Macro", precision_recall_fscore_support(y_test, pred_test, average='macro', labels=range(len(cfg.RELATIONS))))
    print("Micro", precision_recall_fscore_support(y_test, pred_test, average='micro', labels=range(len(cfg.RELATIONS))))

    


def parse_args():
    parser = argparse.ArgumentParser()
    parameters_maxent = OrderedDict()
    
    parser.add_argument(
        "-train_data", default="../data/train_data/",
        help="Standoff_Format gold files"
    )

    parser.add_argument(
        "-test_data", default="../data/test_data/",
        help="Standoff_Format test files"
    )

    parser.add_argument(
        "-dev_data", default="../data/dev_data/",
        help="Standoff_Format dev files"
    )

    opts = parser.parse_args()

    parameters_maxent["train_data"]=opts.train_data
    parameters_maxent["test_data"]=opts.test_data
    parameters_maxent["dev_data"]=opts.dev_data

    return parameters_maxent


def main():
    parameters_maxent = parse_args()


    # train = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=cfg.TRAIN_ARTICLES_PATH)
    # train = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["train_data"])
    # dev = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=cfg.DEV_ARTICLES_PATH)
    # test = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=cfg.TEST_ARTICLES_PATH)
    # test = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["test_data"])
    # dev = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["dev_data"])

    # pickle.dump(test, open('results/pickles/datasets/test.p', 'wb'))
    # pickle.dump(train, open('results/pickles/datasets/train.p', 'wb'))
    # pickle.dump(dev, open('results/pickles/datasets/dev.p', 'wb'))

    try:
        train = pickle.load(open(cfg.Train_Dataset_PICKLE, 'rb'))
    except Exception as e:
        train = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["train_data"])
        pickle.dump(train , open(cfg.Train_Dataset_PICKLE, 'wb'))

    try:
        test = pickle.load(open(cfg.Test_Dataset_PICKLE, 'rb'))
    except Exception as e:
        test = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["test_data"])
        pickle.dump(test, open(cfg.Test_Dataset_PICKLE, 'wb'))


    
    train_df, y_train = train.extract_rel_data()
    test_df, y_test = test.extract_rel_data()

    word_features = ['wm1', 'wbnull', 'wbf', 'wbl', 'wbo', 'bm1f', 'bm1l', 'am2f', 'am2l']
    ent_features = ['et12']
    overlap_features = ['#mb', '#wb']
    chunk_features = ['cphbnull', 'cphbfl', 'cphbf', 'cphbl', 'cphbo', 'cphbm1f', 'cphbm1l', 'cpham2f', 'cpham2l']
    dep_features = ['et1dw1', 'et2dw2', 'h1dw1', 'h2dw2', 'et12SameNP', 'et12SamePP', 'et12SameVP']



    addition = [
        word_features + ent_features + overlap_features + chunk_features + dep_features,
    ]

    for feat in addition:
        print(feat)
        x_train = train.features.tranform(train_df, feat)
        x_test = train.features.tranform(test_df, feat)
        single_run(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()


