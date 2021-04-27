import os
import pickle
from collections import OrderedDict

import pandas as pd
import torch
from transformers import pipeline


if torch.cuda.is_available():
    classifier = pipeline('zero-shot-classification', device=0)
else:
    classifier = pipeline('zero-shot-classification')


thresholds = list(reversed([0.7 + x * 0.01 for x in range(0, 29)] + [0.98 + x * 0.001 for x in range(0, 20)]))


def parse_filename(filename):
    _, company, tweets, direction = filename.split('.')[0].split('-')
    return company, tweets, direction


def nlp(text, candidate_labels, nlp_cache, hypothesis_template="{}."):
    result_dict = OrderedDict.fromkeys(candidate_labels)
    for key in result_dict:
        if text + '__' + key in nlp_cache:
            result_dict[key] = nlp_cache[text + '__' + key]
    remaining_candidate_labels = [c for c in candidate_labels if result_dict[c] is None]
    if len(remaining_candidate_labels) > 0:
        classified = classifier(text, remaining_candidate_labels, hypothesis_template=hypothesis_template,
                                multi_class=True)
        print(classified)
        for key, value in zip(classified['labels'], classified['scores']):
            result_dict[key] = value
            nlp_cache[text + '__' + key] = value
    return OrderedDict(('pred_' + k, v) for k, v in result_dict.items())


def predict(df, df_nli_template, nlp_cache):
    df = df.fillna(0)
    for header in list(df_nli_template):
        hypotheses = df_nli_template[header].dropna().values.tolist()
        if len(hypotheses) == 0:
            break
        hypotheses_labels = ['pred_' + header + '_' + hypothesis for hypothesis in hypotheses]
        df[[label for label in hypotheses_labels]] = df.text.apply(lambda x: pd.Series(nlp(x, hypotheses, nlp_cache)))
    return df


if __name__ == '__main__':

    if not os.path.exists('nlp_cache.pkl'):
        open('nlp_cache.pkl', 'wb+').close()
    nlp_cache_stream = open('nlp_cache.pkl', 'rb')
    try:
        nlp_cache = pickle.load(nlp_cache_stream)
    except EOFError:
        nlp_cache = dict()
    nlp_cache_stream.close()

    if not os.path.exists(os.path.join('data', 'predicted')):
        os.mkdir(os.path.join('data', 'predicted'))

    for filename in os.listdir(os.path.join('data', 'labeled')):
        company, tweets, direction = parse_filename(filename)
        df = pd.read_excel(os.path.join('data', 'labeled', filename), index_col=[0])
        nli_template = 'twcs-{}-nli.xlsx'.format(company)
        df_nli_template = pd.read_excel(os.path.join('data', 'nli-templates', direction, nli_template))
        df = predict(df, df_nli_template, nlp_cache)
        outfile = 'twcs-{}-{}-{}-predicted.xlsx'.format(company, tweets, direction)
        df.to_excel(os.path.join('data', 'predicted', outfile))

    with open('nlp_cache.pkl', 'wb+') as outfile:
        pickle.dump(nlp_cache, outfile)
