import pandas as pd
from stanza.nlp.corenlp import CoreNLPClient
import re

client = CoreNLPClient(server='http://localhost:9000', default_annotators=['ssplit', 'tokenize', 'pos'])

fields = ['Tweet', 'Target', 'Stance']

train = pd.read_csv('/Users/krishna/Desktop/PycharmProjects/cla5/train.csv', engine='python', usecols=fields)
test = pd.read_csv('/Users/krishna/Desktop/PycharmProjects/cla5/test.csv', engine='python', usecols=fields)

def task(train_data, test_data, target):

    print(target)

    trainh = []
    trainj = " "
    traintwts = []

    for index, row in train_data.iterrows():
        if row['Target'] == target:
            l = row['Tweet']
            line = l.lower()
            split = line.split()
            #print(split)
            for string in split:
                if string.startswith('@'):
                    trainh.append(string)
            new_split = [string for string in split if string not in trainh]
            #print(new_split)
            t1 = trainj.join(new_split)
            tweet = re.sub('[^\sa-zA-Z0-9]', '', t1)
            #print(tweet)
            traintwts.append(tweet)

    #print(traintwts)

    train_tweets = pd.DataFrame(traintwts)

    #print(train_tweets)

    output = target[:3]+'_train'+'.csv'

    train_tweets.to_csv(output, index=False, header=None, encoding='utf-8')


    print(target)

    testh = []
    testj = " "
    testtwts = []

    for index, row in test_data.iterrows():
        if row['Target'] == target:
            l = row['Tweet']
            line = l.lower()
            split = line.split()
            #print(split)
            for string in split:
                if string.startswith('@'):
                    testh.append(string)
            new_split = [string for string in split if string not in testh]
            #print(new_split)
            t1 = testj.join(new_split)
            tweet = re.sub('[^\sa-zA-Z0-9]', '', t1)
            #print(tweet)
            testtwts.append(tweet)

    #print(twts)

    test_tweets = pd.DataFrame(testtwts)

    #print(test_tweets)

    output = target[:3]+'_test'+'.csv'

    #test_tweets.to_csv(output, index=False, header=None, encoding='utf-8')

    return 1

targets = []

for index, row in train.iterrows():
    if row['Target'] not in targets:
        targets.append(row['Target'])

for t in targets:
    task(train, test, t)
