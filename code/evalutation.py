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

import itertools


def parse_args():
    parser = argparse.ArgumentParser()
    parameters_maxent = OrderedDict()
    
    parser.add_argument(
        "-gold_data", default="../data/test_data/",
        help="Standoff_Format gold files"
    )

    parser.add_argument(
        "-pred_data", default="output/",
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


    y_gold = []
    y_pred = []
    
    list_of_gold_protocols = gold_data.protocols
    list_of_pred_protocols = []

    for protocol_gold in list_of_gold_protocols:
        for protocol_pred in pred_data.protocols:    
            if protocol_gold.basename == protocol_pred.basename:
                list_of_pred_protocols.append(protocol_pred)
                break



    for protocol_index in range(len(list_of_gold_protocols)):
        gold_protocol = list_of_gold_protocols[protocol_index]
        pred_protocol = list_of_pred_protocols[protocol_index]

        gold_relations = []
        pred_relations = []

        if gold_protocol.relations is not None:
            gold_relations.append(gold_protocol.relations)
        else:
            gold_relations.append([])

        
        if pred_protocol.relations is not None:
            pred_relations.append(pred_protocol.relations)
        else:
            pred_relations.append([])


        gold_y = gold_data.to_idx(list(itertools.chain.from_iterable(gold_relations)))
        pred_y = gold_data.to_idx(list(itertools.chain.from_iterable(pred_relations)))

        
        gold_len = len(gold_y)
        pred_len = len(pred_y)
        no_rel_id = gold_data.rel_label_idx['0']
        

        while len(gold_y) > len(pred_y):
            pred_y.append(no_rel_id)

        while len(pred_y) > len(gold_y):
            gold_y.append(no_rel_id)

        # print(len(gold_y))
        # print(len(pred_y))
        y_gold.extend(gold_y)
        y_pred.extend(pred_y)




    print(len(y_gold), len(y_pred))

    print(classification_report(y_gold, y_pred, target_names=cfg.RELATIONS, labels=range(len(cfg.RELATIONS))))
    print("Macro", precision_recall_fscore_support(y_gold, y_pred, average='macro', labels=range(len(cfg.RELATIONS))))
    print("Micro", precision_recall_fscore_support(y_gold, y_pred, average='micro', labels=range(len(cfg.RELATIONS))))



if __name__ == '__main__':
    main()
