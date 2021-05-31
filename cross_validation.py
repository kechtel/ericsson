import os

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


np.random.seed(1868)

thresholds = list(reversed([0.7 + x * 0.01 for x in range(0, 29)] + [0.98 + x * 0.001 for x in range(0, 20)]))


def parse_filename(filename):
    _, company, tweets, direction, _ = filename.split('.')[0].split('-')
    return company, tweets, direction


def cross_validate_nli_template(df, df_nli_template):
    results = []
    for header in list(df_nli_template):
        hypotheses = df_nli_template[header].dropna().values.tolist()
        if len(hypotheses) == 0:
            continue
        for hypothesis in hypotheses:
            num_positive_instances = len(df[df[header] == 1].index)
            if num_positive_instances >= 3:
                y_true = df[header]
                y_probabilities = df['pred_' + header + '_' + hypothesis]
                skf = StratifiedKFold(n_splits=min(num_positive_instances, 5), shuffle=True)
                threshold_results = []
                for threshold in thresholds:
                    fold_results = []
                    for i, splits in enumerate(skf.split(y_probabilities, y_true)):
                        _, test = splits
                        y_predicted = y_probabilities[test].apply(lambda x: 1 if x > threshold else 0)
                        fold_results.append((threshold,) + evaluate_predictions(y_true[test], y_predicted))
                    threshold_results.append(tuple(map(lambda y: sum(y) / float(len(y)), zip(*fold_results))))
                optimal_threshold = max(threshold_results, key=lambda item: item[1])[0]
                predictions = df['pred_' + header + '_' + hypothesis].apply(lambda x: 1 if x > optimal_threshold else 0)
                evaluation = evaluate_predictions(df[header], predictions)
                results.append((header, hypothesis, optimal_threshold) + evaluation)
    df_results = pd.DataFrame(results, columns=['Header', 'Label', 'Optimal Threshold', 'MCC', 'Accuracy',
                                                'Balanced Accuracy', 'F1', 'Items'])
    return df_results


def evaluate_predictions(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    items = len(y_true[y_true == 1].index)
    return mcc, accuracy, balanced_accuracy, f1, items


if __name__ == '__main__':

    if not os.path.exists('results'):
        os.mkdir('results')

    if not os.path.exists(os.path.join('results', 'nli-cv')):
        os.mkdir(os.path.join('results', 'nli-cv'))

    for filename in os.listdir(os.path.join('data', 'predicted')):
        company, tweets, direction = parse_filename(filename)
        df = pd.read_excel(os.path.join('data', 'predicted', filename), index_col=[0])
        nli_template = 'twcs-{}-nli.xlsx'.format(company)
        df_nli_template = pd.read_excel(os.path.join('data', 'nli-templates', direction, nli_template))
        df_results = cross_validate_nli_template(df, df_nli_template)
        outfile = 'cv_results-{}-{}.xlsx'.format(company, direction)
        writer = pd.ExcelWriter(os.path.join('results', 'nli-cv', outfile), engine='xlsxwriter')
        df_results.to_excel(writer, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        for column in ['E', 'F', 'G', 'H']:
            worksheet.conditional_format('{}2:{}{}'.format(column, column, str(len(df_results.index) + 1)),
                                         {'type': '3_color_scale'})
        writer.save()
