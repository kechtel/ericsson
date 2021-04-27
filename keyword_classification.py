import os

import pandas as pd
from ericsson.cross_validation import evaluate_predictions


def parse_filename(filename):
    _, company, direction, type = filename.split('.')[0].split('-')
    return company, direction, type


def predict(text, keyword):
    return 1 if keyword.lower() in text.lower() else 0


if __name__ == '__main__':

    if not os.path.exists('results'):
        os.mkdir('results')

    if not os.path.exists(os.path.join('results', 'keyword-classification')):
        os.mkdir(os.path.join('results', 'keyword-classification'))

    for filename in os.listdir(os.path.join('data', 'topics-activities')):
        company, direction, type = parse_filename(filename)
        df_mapping = pd.read_excel(os.path.join('data', 'topics-activities', filename))
        df = None
        mapping = None
        if type == 'topics':
            df = pd.read_excel(os.path.join('data', 'labeled', 'twcs-{}-200-inbound.xlsx'.format(company)))
            mapping = dict(zip(df_mapping.Topic, df_mapping.Keyword))
        else:
            df = pd.read_excel(os.path.join('data', 'labeled', 'twcs-{}-100-outbound.xlsx'.format(company)))
            mapping = dict(zip(df_mapping.Activity, df_mapping.Keyword))
        df = df.fillna(0)
        results = []
        for item, keyword in mapping.items():
            predictions = df['text'].apply(lambda x: predict(x, keyword))
            results.append((item, keyword) + evaluate_predictions(df[item], predictions))
        df_results = pd.DataFrame(results, columns=['Topic/Activity', 'Keyword', 'MCC',
                                                    'Accuracy', 'Balanced Accuracy', 'F1', 'Items'])
        outfile = 'keyword_results-{}-{}.xlsx'.format(company, direction)
        writer = pd.ExcelWriter(os.path.join('results', 'keyword-classification', outfile), engine='xlsxwriter')
        df_results.to_excel(writer, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        for column in ['D', 'E', 'F', 'G']:
            worksheet.conditional_format('{}2:{}{}'.format(column, column, str(len(df_results.index) + 1)),
                                         {'type': '3_color_scale'})
        writer.save()
