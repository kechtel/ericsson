import csv
import os

import fasttext
import networkx as nx
import pandas as pd
from spellchecker import SpellChecker

spell = SpellChecker()

model = fasttext.load_model('lid.176.ftz')


def filter_outbound_tweets_by_company(df, companies):
    return df[(df['inbound'] == True) | (df['author_id'].isin(companies))]


def add_main_tweet_id(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.tz_localize(None)
    df_responses = df[df['in_response_to_tweet_id'].notnull()]
    g = nx.from_pandas_edgelist(df_responses, source='in_response_to_tweet_id', target='tweet_id',
                                create_using=nx.DiGraph)
    roots = {n for n, d in g.in_degree() if d == 0}
    d = {}
    components = nx.weakly_connected_components(g)
    for i, comp in enumerate(components):
        comp_root = next(root for root in roots if root in comp)
        d.update(dict.fromkeys(comp, comp_root))
    df['main_tweet_id'] = df['tweet_id'].map(d)
    df['main_tweet_id'].fillna(df['in_response_to_tweet_id'], inplace=True)
    df['main_tweet_id'].fillna(df['tweet_id'], inplace=True)
    df = df.sort_values(by=['main_tweet_id', 'created_at'])
    df = df[['main_tweet_id', 'tweet_id', 'in_response_to_tweet_id', 'response_tweet_id', 'created_at', 'author_id',
             'inbound', 'text']]
    return df


def add_company(df):
    df_companies = df[df['inbound'] == False][['main_tweet_id', 'author_id']].drop_duplicates()
    df_companies.rename({'author_id': 'company'}, axis=1, inplace=True)
    # This mapping might not be unique for every tweet since there are conversations with multiple companies involved
    # By removing the conversations with multiple companies entirely we account for this issue
    df = df.merge(df_companies, on='main_tweet_id', suffixes=('', ''))
    return df


def remove_conversations_with_multiple_companies(df):
    return df.groupby(by=['main_tweet_id']).filter(lambda g: g.company.nunique() == 1)


def remove_non_english_tweets(df):
    return df[df.apply(lambda x: model.predict(str(x['text']).replace('\n', ''))[0][0] == '__label__en', axis=1)]


def remove_non_conversational_tweets(df):
    return df.groupby(by=['main_tweet_id']).filter(lambda g: g.main_tweet_id.count() > 1)


def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


def correct_spellings_inbound(df):
    df['text'] = df.apply(lambda x: correct_spellings(x['text']) if x['inbound'] == True else x['text'], axis=1)
    return df


if __name__ == '__main__':
    companies = ['AmazonHelp', 'AppleSupport', 'SpotifyCares']
    df = pd.read_csv('twcs.csv', quoting=csv.QUOTE_ALL)
    df = df.replace('\n', ' ', regex=True)
    df = filter_outbound_tweets_by_company(df, companies)
    df = df.sample(frac=0.02)
    df = add_main_tweet_id(df)
    df = add_company(df)
    df = remove_conversations_with_multiple_companies(df)
    df = remove_non_english_tweets(df)
    df = remove_non_conversational_tweets(df)
    df = correct_spellings_inbound(df)

    if not os.path.exists('data'):
        os.mkdir('data')

    if not os.path.exists(os.path.join('data', 'preprocessed')):
        os.mkdir(os.path.join('data', 'preprocessed'))

    for company in companies:
        df_company = df[df['company'] == company]
        filename = 'twcs-{}-preprocessed-{}.xlsx'.format(company, str(len(df_company.index)))
        df_company.to_excel(os.path.join('data', 'preprocessed', filename))
