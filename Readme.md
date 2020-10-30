# WNUT 2020: Relation Extraction 


WNUT 2020 shared task is designed on the wet lab protocol data. Wet Lab protocols are bascially a collection of steps from different lab procedures. They are noisy, dense, and domain-specific. Automatic or semi-automatic conversion of protocols into machine-readable format benefits medical and biological research. In this task, participants are invited for extracting relations from the entities of these lab protocols. Note that these protocols are written by researchers and lab technicians from all over the world, some of which may contain non-standard language or spelling errors. 

 All of the protocols were collected from [protocols.io](https://www.protocols.io/) using their public APIs. The full protocol-dump is also available as json format in their [github repository](https://github.com/protocolsio/protocols). For this shared task, we provide the annotation of 615 protocols. The [BRAT](https://brat.nlplab.org/examples.html) styled annotated protocols can be visulalized at: http://bit.ly/WNUT2020. Below is a sample of the input data:

![nCoV-2019 sequencing protocol](./covid-data.png?raw=true "Title")

## Shared-task Organizers

- [Jeniya Tabassum](https://sites.google.com/site/jeniyatabassum/) (Ohio State University)
- [Wei Xu](https://cocoxu.github.io/) (Ohio State University → Georgia Tech)
- [Alan Ritter](http://aritter.github.io/) (Ohio State University → Georgia Tech)

## Shared-task Details

- [Participat's Registration](https://docs.google.com/forms/d/e/1FAIpQLSem8kMoPeHJKa78xFSat1Iy8FWLFXB6zDG3qiRl92h3kahBUg/viewform) 
- [Task' Website](http://noisy-text.github.io/2020/wlp-task.html) 

## Data

Our Wet Lab Protocol (WLP) dataset consists of 615 unique protocols from the 623 protocols of [Kulkarni et al. (2018)](https://cocoxu.github.io/publications/NAACL_2018_wet_lab_protocols.pdf). It excludes the following 8 duplicated protocols: 

- protocol 45 (duplicate of protocol 441)
- protocol 459 (duplicate of protocol 310)
- protocol 464 (duplicate of protocol 46)
- protocol 480 (duplicate of protocol 473)
- protocol 482 (duplicate of protocol 474)
- protocol 483 (duplicate of protocol 475)
- protocol 484 (duplicate of protocol 476)
- protocol 621 (duplicate of protocol 570)

After discarding the duplicate protocols, the remaining 615 unique protocols are re-annoated in brat with 3 annotators with 0.75 inter-annotator agreement, measured by span-level Cohen’s Kappa. Our annotators added the missing entity-relations and also corrected the incosistencies. The updated dataset is provided in the [data directory](./data/Readme.md) in StandOff format. The data is divided in 3 sub-directories as below:

1) [train_data](./data/train_data/): 370 protocols with 8444 sentences
2) [dev_data](./data/dev_data/): 122 protocols  with 2839 sentences
3) [test_data](./data/test_data/): 123 protocols  with 2813 sentences


**We will release a new test set on September 9, 2020 for the offical evaluation of the Relation Extraction task. Please sign up [here](https://forms.gle/HtnNezLTgkuj4DyQ8) to receive the data in email. We will use a [mailing list](https://groups.google.com/forum/#!forum/wnut2020-sharedtask) for annoucements and discussions.**

## Timeline 

- Data available: June 8
- Evaluation window: Sep 9 - Sep 15
- System Description Papers Submitted: Sep 18
- Papers Reviewed: Sep 28
- Papers Camera-ready: Oct 8
- Workshop Day: Nov 11




## Baselines

We have provided a feature based [Logistic Regression Model](./code/Readme.md#-The-baseline-relation-extraction-model) for the Relation Extraction Task.


## Evaluation

The participants are required to produce predictions on the protocols as [StandOff format](./data/Readme.md#-The-standoff-format:), which will be compared with the gold data using the [evalution script](./code/Readme.md#-The-evaluation-system).

## Directory Structure for Important Files and Folders
```
.
├── code
│   ├── rel_classifier.py
│   ├── predict_relations.py
│   └── evalutation.py
└── data
    ├── dev_data
    ├── test_data
    └── train_data
```
## Relevant Paper:
 Paper about the shared task:
 
   
	@inproceedings{tabassum2020,
	  author={Tabassum, Jeniya and Lee, Sydney and Xu, Wei and Ritter, Alan },
	  title={{WNUT-2020 Task 1 Overview: Extracting Entities and Relations from Wet Lab Protocols}},
	  booktitle={Proceedings of EMNLP 2020 Workshop on Noisy User-generated Text {(WNUT)}},
	  year = {2020}
	} 

  

 Paper about the dataset:
   
	@inproceedings{kulkarni2018wetlab,
	  author     = {Kulkarni, Chaitanya and Xu, Wei and Ritter, Alan and Machiraju, Raghu},
	  title      = {An Annotated Corpus for Machine Reading of Instructions in Wet Lab Protocols},
	  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)},
	  year       = {2018}
	} 

  

  
