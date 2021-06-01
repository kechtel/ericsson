# ERICSSON - Event log constRuctIon from CuStomer Service cOnversatioNs

![example workflow](https://github.com/kechtel/ericsson/actions/workflows/python-app.yml/badge.svg)

This repository contains the source code and the online appendix for our paper entitled "Event Log Construction from Customer Service Conversations using Natural Language Inference". 
For a more comprehensive overview of the approach and the steps invoked in the scripts listed below we refer to the paper.


## Installation

To checkout the repository and to create a new Python virtual environment execute the following commands:

```
git clone https://github.com/kechtel/ericsson
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Preprocessing 

The file `preprocessing.py` extracts the Twitter conversations from AmazonHelp, AppleSupport, and SpotifyCares from the "Customer Support on Twitter" dataset (https://www.kaggle.com/thoughtvector/customer-support-on-twitter).
It furthermore applies some basic preprocessing steps, such as adding an identifier for each conversation, removing conversations that involve multiple companies, removing non-english Tweets, removing non-conversational Tweets, and spelling correction.
The resulting preprocessed dataframes are saved in the path `data/preprocessed`.
To run the file, execute the following command:

```
python3 preprocessing.py
```

## Natural Language Inference

The file `nli.py` requires labeled inbound and outbound Tweets in the path `data/labeled` for each of the companies and a list of potential NLI hypotheses in the path `data/nli-templates` for each of the topics and process actions to extract from the conversations.
For each combination of hypothesis and topic/process action, the file calculates the probability of belonging to the topic/process action for each labeled Tweet using NLI.
The resulting dataframes are saved in the path `data/predicted`.
To run the file, execute the following command:

```
python3 nli.py
```

## Cross Validation

The file `cross_validation.py` determines the optimal binary classification decision threshold using cross-validation for each of the NLI hypotheses evaluated in the previous step and reports binary classification evaluation metrics for each of the hypotheses.
The results are saved in the path `results/nli-cv`.
To run the file, execute the following command:

```
python3 cross_validation.py
```

## Keyword Classification

The file `keyword_classification.py` provides an alternative approach to NLI by simply searching the Tweets for a specific keyword that describes a topic or process activity.
It requires the keywords for the inbound and outbound Tweets for each of the companies in the path `data/topics-activities`.
The results are saved in the path `results/keyword-classification`.
To run the file, execute the following command:

```
python3 keyword_classification.py
```

## Event Log Construction

The file `event_log_construction.py` converts the conversations in the path `data/preprocessed` to an XES event log using NLI.
It requires the NLI hypothesis and decision threshold for each topic and process activity for each of the companies in the path `data/topics-activities`.
The resulting XES event logs are saved in the path `xes`.
To run the file, execute the following command:

```
python3 event_log_construction.py
```

## Process Mining

The file `process_mining.py` discovers and visualizes process models from the conversations using PM4Py's Alpha Miner, Heuristics Miner, Inductive Miner, and Directly Follows Graph.
It requires the XES log obtained by converting the conversations in the path `xes` and filters the log by traces that start with a message of the customer and futhermore applies PM4Py's auto variants filter with a decreasing factor of 0.7.
The resulting visualized process models are saved in the path `results/process-discovery`.
To run the file, execute the following command:

```
python3 process_mining.py
```
