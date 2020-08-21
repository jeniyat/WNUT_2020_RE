# The baseline relation extraction model:

We provided a simple logistic regression model using contextual, lexical and gazetter features. The following script contains the baseline Feature Based CRF model:

```
      ./rel_classifier.py

```
### How to run baseline model:

To run and get the prediction on the test data, you need download the [stanford dependecy parser](https://mega.nz/file/HcgEyIJI#keL_1DL00BcLn_DwsswZuWrOuyJaJ1T9OPtJQpjjMXs) and unzip it  under the`./feature_engineering/` directory. 

The classifier can be trained and test using the  `./rel_classifier.py` script with the following parameters:

1) The `<location of StandOff format train data>` in `-train_data` parameter, and 
2) The `<location of StandOff format test data>` in the `-test_data` parameter


```
python rel_classifier.py  -train_data "../../data/train_data/" -test_data "../../data/test_data/"
```

### How to save prediction from baseline model:

To run and get the prediction on the test data, you need download the [stanford dependecy parser](https://mega.nz/file/HcgEyIJI#keL_1DL00BcLn_DwsswZuWrOuyJaJ1T9OPtJQpjjMXs) and unzip it  under the`./feature_engineering/` directory. 

The classifier can be trained and the predictions on the test data can be saves using the  `./predict_relations.py` script with the following parameters:

1) The `<location of StandOff format train data>` in `-train_data` parameter, 
2) The `<location of StandOff format test data>` in the `-test_data` parameter, and 
3) The `<location of StandOff format predicted output>` in the `-op_dir` parameter


```
python predict_relations.py  -train_data "../../data/train_data/" -test_data "../../data/test_data/" -op_dir "output/"
```



# The evaluation system:


The following script should be used to evaluate the performance of the model predictions:
  
      ./evalutation.py


The participants are required to produce entity sequence for each sentence and submit the predictions as either [StandOff format](../../data/Readme.md##-The-standoff-format:).


The evaluation script takes `<location of gold data>` and `<location of predicted data>` as input, and then outputs the detailed perfromance of the NER tagger. 

For example, below is the detailed performance of the provided maximum entropy model.

```
                    precision    recall  f1-score   support
 
          Acts-on       0.85      0.86      0.86      2951
             Site       0.72      0.70      0.71      1077
          Creates       0.00      0.00      0.00         1
            Using       0.68      0.54      0.60       781
            Count       0.85      0.76      0.80        99
Measure-Type-Link       0.73      0.82      0.77       106
 Coreference-Link       0.50      0.27      0.35        84
         Mod-Link       0.83      0.79      0.81      1523
          Setting       0.88      0.83      0.85      1695
          Measure       0.89      0.87      0.88      1923
          Meronym       0.74      0.29      0.42       441
               Or       0.46      0.17      0.24       174
          Of-Type       0.65      0.50      0.56        22
 
        micro avg       0.83      0.77      0.80     10877
        macro avg       0.68      0.57      0.61     10877
     weighted avg       0.82      0.77      0.79     10877
 
Macro (0.6753072258720733, 0.5697930611906763, 0.6053845316507502, None)
Micro (0.8267918932278794, 0.7688700928564861, 0.796779725609756, None)

``` 
    


### How to run evaluation:

To run the evalutaion script, you need to provide it with-

1) The `<location of StandOff format gold data>` in `-gold_data` parameter, and 
2) The `<location of StandOff format predicted data>` in the `-pred_data` parameter

The gold data and the prediction data must be formated as the StandOff format or the CoNLL fomrat.

```
python evaluation.py  -gold_data "../data/test_data/" -pred_data "RE_Outputs/"

```

### Import evalution function inside another script:

The evalutaion funciton is utilizing `classification_report` and `precision_recall_fscore_supportcan` from `sklearn.metrics` and can be from other python scripts as below:

```

import evaluation


evaluation.find_perfomance(gold_data_location="../data/test_data", pred_data_location="outputs")

```



# Requirements:

All the codes are written in python 3. The following python modules are required by the scripts provided in this code-base.


```
sklearn
nltk
codecs
io
re
os
sys
glob
shutil
argparse
gensim
stanford-core-nlp

```
