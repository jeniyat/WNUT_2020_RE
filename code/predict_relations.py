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

import shutil

import evaluation


def single_run(train, x_train, y_train, test, x_test, y_test, output_dir):
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=8)
    try:
        print("now loading the pickled model from: results/pickles/models/", )
        model = pickle.load(open('results/pickles/models/LR_rel_classifier.p', 'rb'))
    except Exception as e:
        print("could not load the saved model. now training the model.")
        
        model.fit(x_train, y_train)

        print("now dumping the model")
        pickle.dump(model , open('LR_rel_classifier.p', 'wb'))
    
    list_of_test_protocols = test.protocols
    last_index = 0
    

    list_of_test_protocols = test.protocols
    last_index = 0
    
    for protocol in list_of_test_protocols:
        
        protocol_rel_len = len(protocol.relations)
        
        
        
        x_test_ = x_test[last_index:last_index+protocol_rel_len]
        
        pred_x = model.predict(x_test_)
        last_index= protocol_rel_len

        protocol.write_rels(pred_x, test.rel_label_idx, write_copy=output_dir+protocol.basename)
    print("predictions are saved in :", output_dir)
        
        




def parse_args():
    parser = argparse.ArgumentParser()
    parameters_maxent = OrderedDict()
    
    parser.add_argument(
        "-train_data", default="../data/train_data_small/",
        help="Standoff_Format gold files"
    )

    parser.add_argument(
        "-test_data", default="../data/test_data_small/",
        help="Standoff_Format test files"
    )

    parser.add_argument(
        "-op_dir", default="output/",
        help="Folder to save the perdictions on test data"
    )

    opts = parser.parse_args()

    parameters_maxent["train_data"]=opts.train_data
    parameters_maxent["test_data"]=opts.test_data
    parameters_maxent["output_directory"]=opts.op_dir

    return parameters_maxent


def main():
    parameters_maxent = parse_args()
    # print(parameters_maxent)

    output_dir = parameters_maxent["output_directory"]

    train = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["train_data"])
    test = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=parameters_maxent["test_data"])

    shutil.rmtree(output_dir, ignore_errors=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        # print(feat)
        x_train = train.features.tranform(train_df, feat)
        x_test = train.features.tranform(test_df, feat)
        single_run(train, x_train, y_train, test, x_test, y_test, output_dir)

    evaluation.find_perfomance(gold_data_location=parameters_maxent["test_data"], pred_data_location=output_dir)


if __name__ == '__main__':
    main()
