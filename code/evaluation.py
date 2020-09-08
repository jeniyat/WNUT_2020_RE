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


def get_key( protocol_name, sent_idx, arg1_tag, arg2_tag):
    arg1_tag_start = arg1_tag.start
    arg1_tag_end = arg1_tag.end

    arg2_tag_start = arg2_tag.start
    arg2_tag_end = arg2_tag.end

    key = protocol_name+"---"+str(sent_idx)+"---"+str(arg1_tag_start)+"---"+str(arg1_tag_end)+"---"+str(arg2_tag_start)+"---"+str(arg2_tag_end)

    return key


def main():
    parameters_maxent = parse_args()

    # gold_data = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["gold_data"])
    # pred_data = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["pred_data"])

    # pickle.dump(pred_data , open('pred_data.p', 'wb'))
    # pickle.dump(gold_data , open(cfg.Test_Dataset_PICKLE, 'wb'))
    try:
        gold_data = pickle.load(open(cfg.Test_Dataset_PICKLE, 'rb'))
    except Exception as e:
        gold_data = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["gold_data"])
        pickle.dump(gold_data , open(cfg.Test_Dataset_PICKLE, 'wb'))

    try:
        pred_data = pickle.load(open('pred_data.p', 'rb'))
    except Exception as e:
        pred_data = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["pred_data"])
        pickle.dump(pred_data ,open('pred_data.p', 'wb'))


    y_gold = []
    y_pred = []
    
    
    gold_data_protocol_label_dict = {}

    pred_data_protocol_label_dict = {}

    list_of_pred_protocols = pred_data.protocols
    list_of_gold_protocols = gold_data.protocols

    

    for protocol_gold in list_of_gold_protocols:
        for relation in protocol_gold.relations:
            
            protocol_name = relation.p.basename
            sent_number  = relation.sent_idx

            arg1_tag = relation.arg1_tag
            arg2_tag = relation.arg2_tag


            relation_label = relation.label
            


            key_ = get_key(protocol_name, sent_number, arg1_tag, arg2_tag)

            
            gold_data_protocol_label_dict[key_] = relation_label
    

    for protocol_pred in list_of_pred_protocols:
        for relation in protocol_pred.relations:
            
            protocol_name = relation.p.basename
            sent_number  = relation.sent_idx

            arg1_tag = relation.arg1_tag
            arg2_tag = relation.arg2_tag


            relation_label = relation.label
            

            key_ = get_key(protocol_name, sent_number, arg1_tag, arg2_tag)

            
            pred_data_protocol_label_dict[key_] = relation_label


    for key_ in pred_data_protocol_label_dict:
        if key_ in gold_data_protocol_label_dict:
            pred_label = pred_data_protocol_label_dict[key_]
            gold_label = gold_data_protocol_label_dict[key_]
        else:
            gold_label = cfg.NEG_REL_LABEL

        gold_index = gold_data.rel_label_idx[gold_label]
        pred_index = gold_data.rel_label_idx[pred_label]

        y_gold.append(gold_index)
        y_pred.append(pred_index)

    for key_ in gold_data_protocol_label_dict:
        if key_ not in pred_data_protocol_label_dict:
            pred_label = cfg.NEG_REL_LABEL
            gold_label = gold_data_protocol_label_dict[key_]

            gold_index = gold_data.rel_label_idx[gold_label]
            pred_index = gold_data.rel_label_idx[pred_label]

            y_gold.append(gold_index)
            y_pred.append(pred_index)

    print(len(y_gold))
    print(len(y_pred))
    print(classification_report(y_gold, y_pred, target_names=cfg.RELATIONS, labels=range(len(cfg.RELATIONS))))
    print("Macro", precision_recall_fscore_support(y_gold, y_pred, average='macro', labels=range(len(cfg.RELATIONS))))
    print("Micro", precision_recall_fscore_support(y_gold, y_pred, average='micro', labels=range(len(cfg.RELATIONS))))


            
    

    

    
    


def find_perfomance(gold_data_location, pred_data_location):
    gold_data = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=gold_data_location)
    pred_data = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=gold_data_location)

    y_gold = []
    y_pred = []
    
    
    gold_data_protocol_label_dict = {}

    pred_data_protocol_label_dict = {}

    list_of_pred_protocols = pred_data.protocols
    list_of_gold_protocols = gold_data.protocols

    

    for protocol_gold in list_of_gold_protocols:
        for relation in protocol_gold.relations:
            
            protocol_name = relation.p.basename
            sent_number  = relation.sent_idx

            arg1_tag = relation.arg1_tag
            arg2_tag = relation.arg2_tag


            relation_label = relation.label
            


            key_ = get_key(protocol_name, sent_number, arg1_tag, arg2_tag)

            
            gold_data_protocol_label_dict[key_] = relation_label
    

    for protocol_pred in list_of_pred_protocols:
        for relation in protocol_pred.relations:
            
            protocol_name = relation.p.basename
            sent_number  = relation.sent_idx

            arg1_tag = relation.arg1_tag
            arg2_tag = relation.arg2_tag


            relation_label = relation.label
            

            key_ = get_key(protocol_name, sent_number, arg1_tag, arg2_tag)

            
            pred_data_protocol_label_dict[key_] = relation_label


    for key_ in pred_data_protocol_label_dict:
        if key_ in gold_data_protocol_label_dict:
            pred_label = pred_data_protocol_label_dict[key_]
            gold_label = gold_data_protocol_label_dict[key_]
        else:
            gold_label = cfg.NEG_REL_LABEL

        gold_index = gold_data.rel_label_idx[gold_label]
        pred_index = gold_data.rel_label_idx[pred_label]

        y_gold.append(gold_index)
        y_pred.append(pred_index)

    for key_ in gold_data_protocol_label_dict:
        if key_ not in pred_data_protocol_label_dict:
            pred_label = cfg.NEG_REL_LABEL
            gold_label = gold_data_protocol_label_dict[key_]

            gold_index = gold_data.rel_label_idx[gold_label]
            pred_index = gold_data.rel_label_idx[pred_label]

            y_gold.append(gold_index)
            y_pred.append(pred_index)

    print(len(y_gold))
    print(len(y_pred))
    print(classification_report(y_gold, y_pred, target_names=cfg.RELATIONS, labels=range(len(cfg.RELATIONS))))
    print("Macro", precision_recall_fscore_support(y_gold, y_pred, average='macro', labels=range(len(cfg.RELATIONS))))
    print("Micro", precision_recall_fscore_support(y_gold, y_pred, average='micro', labels=range(len(cfg.RELATIONS))))

    



if __name__ == '__main__':
    main()
