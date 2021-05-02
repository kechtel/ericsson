import os
import pickle

import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.util import dataframe_utils

from nli import nlp


def parse_filename(filename):
    _, company, _, tweets = filename.split('.')[0].split('-')
    return company, tweets


def append_activities_and_topics(row, df_activity_mappings, df_topic_mappings, nlp_cache):
    activities_topics = []
    if row['author_id'] == row['company']:
        nlp_results = nlp(row['text'], list(df_activity_mappings['Label']), nlp_cache)
        for label, probability in nlp_results.items():
            label = label[5:]
            if probability > list(df_activity_mappings[df_activity_mappings['Label'] == label]['Optimal Threshold'])[0]:
                activities_topics.append(
                    list(df_activity_mappings[df_activity_mappings['Label'] == label]['Activity'])[0])
    elif row['tweet_id'] == row['main_tweet_id']:
        nlp_results = nlp(row['text'], list(df_topic_mappings['Label']), nlp_cache)
        for label, probability in nlp_results.items():
            label = label[5:]
            if probability > list(df_topic_mappings[df_topic_mappings['Label'] == label]['Optimal Threshold'])[0]:
                activities_topics.append(
                    list(df_topic_mappings[df_topic_mappings['Label'] == label]['Topic'])[0])
    row['activities_topics'] = activities_topics
    return row


def rename_df_for_xes(df):
    mapping = {
        'main_tweet_id': 'case:concept:name',
        'created_at': 'time:timestamp',
        'author_id': 'org:resource',
        'activities_topics': 'concept:name'
    }
    return df.rename(mapping, axis=1)


def to_event_log(df_conversations, df_activity_mappings, df_topic_mappings, nlp_cache):
    df_conversations = df_conversations.drop(df_conversations.columns[[0]], axis=1)
    df_conversations = df_conversations.drop(['in_response_to_tweet_id', 'response_tweet_id', 'inbound'], axis=1)
    df_event_log = df_conversations.apply(lambda row: append_activities_and_topics(row, df_activity_mappings, df_topic_mappings, nlp_cache), axis=1)

    with open('nlp_cache.pkl', 'wb+') as outfile:
        pickle.dump(nlp_cache, outfile)

    df_event_log['text'] = df_event_log['text'].map(lambda x: x.replace('"', r'\''))
    df_event_log['text'] = df_event_log['text'].map(lambda x: x.replace('<', r'&lt;'))
    df_event_log['text'] = df_event_log['text'].map(lambda x: x.replace('>', r'&gt;'))
    df_event_log['text'] = df_event_log['text'].map(lambda x: x.replace('& amp;', r'&amp;'))

    df_event_log = rename_df_for_xes(df_event_log)
    df_event_log = df_event_log.explode('concept:name')
    df_event_log = df_event_log[df_event_log['concept:name'].notna()]

    event_log = dataframe_utils.convert_timestamp_columns_in_df(df_event_log)
    event_log = log_converter.apply(event_log)
    return event_log


if __name__ == '__main__':

    if not os.path.exists('nlp_cache.pkl'):
        open('nlp_cache.pkl', 'wb+').close()
    nlp_cache_stream = open('nlp_cache.pkl', 'rb')
    try:
        nlp_cache = pickle.load(nlp_cache_stream)
    except EOFError:
        nlp_cache = dict()
    nlp_cache_stream.close()

    if not os.path.exists('xes'):
        os.mkdir('xes')

    for filename in os.listdir(os.path.join('data', 'preprocessed')):
        company, tweets = parse_filename(filename)
        activity_mappings_file = os.path.join('data', 'topics-activities', 'twcs-' + company + '-outbound-activities.xlsx')
        topic_mappings_file = os.path.join('data', 'topics-activities', 'twcs-' + company + '-inbound-topics.xlsx')
        conversations_file = os.path.join('data', 'preprocessed', filename)

        df_activity_mappings = pd.read_excel(activity_mappings_file)
        df_topic_mappings = pd.read_excel(topic_mappings_file)
        df_conversations = pd.read_excel(conversations_file)

        event_log = to_event_log(df_conversations, df_activity_mappings, df_topic_mappings, nlp_cache)

        xes_exporter.apply(event_log, 'xes/' + 'twcs-' + company + '-' + tweets + '.xes')
