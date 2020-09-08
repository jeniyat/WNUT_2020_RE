The `Standoff_Format` folder contains the surprised test data for the WNUT-2020 shared task on "Relation Extraction from Wet Lab Protocols". 

Currently the data of this folder contain any gold labels for the named entities only. The gold labels of the realations among these entities will be released after the evalauation period closes.



# Submission of System Output
 
You need to submit your model prediction on these new data by  **September 15, 2020 (AoE)**, with a brief description (<= 280 characters) of your model  using the [output submission the form](https://forms.gle/EabVTq8afgaaJxEW9). 


 
## Submission Instructions

You are required to submit your model predictions in a zip file. The name of the zipped file must be in the following format:

```
	<team_name>.zip  
	[e.g., ‘OSU_NLP’ team must submit the predictions in ‘OSU_NLP.zip’]
```

The zipped file should contain the predictions on these 111 protocols. You **must** submit your predictions in [**standoff format**](../../data#the-standoff-format), and utilize the following directory structure: 

```
	OSU_NLP/
	  ├── protocol_0623.ann
	  ├── protocol_0623.txt
	  ├── protocol_0624.ann
	  ├── protocol_0624.txt
	  ├── protocol_0625.ann
	  ├── protocol_0625.txt
	  ├── protocol_0626.ann
	  ├── ...
```

