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



# def single_run(x_train, y_train, x_test, y_test, x_dev, y_dev):
#     model = LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=8)

#     model.fit(x_train, y_train)

#     print("Results on test set: ")
#     pred_test = model.predict(x_test)

#     print(classification_report(y_test, pred_test, target_names=cfg.RELATIONS, labels=range(len(cfg.RELATIONS))))
#     print("Macro", precision_recall_fscore_support(y_test, pred_test, average='macro', labels=range(len(cfg.RELATIONS))))
#     print("Micro", precision_recall_fscore_support(y_test, pred_test, average='micro', labels=range(len(cfg.RELATIONS))))

#     print("Results on dev set: ")
#     pred_dev = model.predict(x_dev)

#     print(classification_report(y_dev, pred_dev, target_names=cfg.RELATIONS, labels=range(len(cfg.RELATIONS))))
#     print("Macro", precision_recall_fscore_support(y_dev, pred_dev, average='macro', labels=range(len(cfg.RELATIONS))))
#     print("Micro", precision_recall_fscore_support(y_dev, pred_dev, average='micro', labels=range(len(cfg.RELATIONS))))


def parse_args():
    parser = argparse.ArgumentParser()
    parameters_maxent = OrderedDict()
    
    parser.add_argument(
        "-gold_data", default="../data/test_data/",
        help="Standoff_Format gold files"
    )

    parser.add_argument(
        "-pred_data", default="../data/test_data/",
        help="Standoff_Format prediction files"
    )

    opts = parser.parse_args()

    parameters_maxent["gold_data"]=opts.gold_data
    parameters_maxent["pred_data"]=opts.pred_data

    return parameters_maxent

def main():
    parameters_maxent = parse_args()

    gold_data = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["gold_data"])
    pred_data = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["pred_data"])

    gold_df, y_gold = gold_data.extract_rel_data()
    pred_df, y_pred = pred_data.extract_rel_data()

    print(classification_report(y_gold, y_pred, target_names=cfg.RELATIONS, labels=range(len(cfg.RELATIONS))))
    print("Macro", precision_recall_fscore_support(y_gold, y_pred, average='macro', labels=range(len(cfg.RELATIONS))))
    print("Micro", precision_recall_fscore_support(y_gold, y_pred, average='micro', labels=range(len(cfg.RELATIONS))))



if __name__ == '__main__':
    main()
