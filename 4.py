import pandas as pd
#from stanza.nlp.corenlp import CoreNLPClient
from sklearn.feature_extraction.text import CountVectorizer
import re

#client = CoreNLPClient(server='http://localhost:9000', default_annotators=['ssplit', 'tokenize', 'pos'])

fields = ['Tweet', 'Target', 'Stance']
train = pd.read_csv('/Users/krishna/Desktop/PycharmProjects/cla5/train.csv', engine='python', usecols=fields)
test = pd.read_csv('/Users/krishna/Desktop/PycharmProjects/cla5/test.csv', engine='python', usecols=fields)

def task4(train_data, test_data, target):

    print(target)

    twts =[]
    s = []
    j = " "

    for index, row in train_data.iterrows():
        if row['Target'] == target:
            l = row['Tweet']
            line = l.lower()
            split = line.split()
            for string in split:
                if string.startswith('@'):
                    s.append(string)
            new_split = [string for string in split if string not in s]
            t1 = j.join(new_split)
            tweet = re.sub('[^\sa-zA-Z0-9]', '', t1)
            twts.append(tweet)

    #print(twts)

    #annotated = client.annotate(str(twts))

    bag_of_words = []

    for twt in twts:
        words = twt.split()
        bag_of_words.append(words)

    #print(bag_of_words)

    word_list = []

    for sublist in bag_of_words:
        for item in sublist:
            word_list.append(item)

    #print(word_list)

    features = list(set(word_list))

    #print(features)
    print(len(features))

    cv = CountVectorizer(vocabulary=features)

    train_tweets = []
    train_stance = []

    for index, row in train_data.iterrows():
        if row['Target'] == target:
            train_stance.append(row['Stance'])

    traintweets_df = pd.DataFrame({'Tweet':twts})
    trainstance_df = pd.DataFrame({'Stance':train_stance})

    train_vectors = cv.fit_transform(traintweets_df['Tweet'])
    train_vectors1 = train_vectors.toarray()

    df1 = pd.DataFrame(train_vectors1, columns=cv.get_feature_names())
    #print(df1.head())

    df2 = pd.concat([df1, trainstance_df], axis=1)

    #print(df2)

    output = 'all_train_'+target[:3]+'.csv'

    #df2.to_csv(output, index=False, header=None, encoding='utf-8')

    test_tweets = []
    test_stance = []

    ts = []
    tj = " "

    for index, row in test_data.iterrows():
        if row['Target'] == target:
            l = row['Tweet']
            line = l.lower()
            split = line.split()
            for string in split:
                if string.startswith('@'):
                    ts.append(string)
            new_split = [string for string in split if string not in ts]
            t1 = tj.join(new_split)
            tweet = re.sub('[^\sa-zA-Z0-9]', '', t1)
            test_tweets.append(tweet)
            test_stance.append(row['Stance'])

    testtweets_df = pd.DataFrame({'Tweet':test_tweets})
    teststance_df = pd.DataFrame({'Stance':test_stance})

    test_vectors = cv.fit_transform(testtweets_df['Tweet'])
    test_vectors1 = test_vectors.toarray()

    df3 = pd.DataFrame(test_vectors1, columns=cv.get_feature_names())
    #print(df3.head())

    df4 = pd.concat([df3, teststance_df], axis=1)

    #print(df4)

    output = 'all_test_'+target[:3]+'.csv'

    #df4.to_csv(output, index=False, header=None, encoding='utf-8')

    return 1

targets = []

for index, row in train.iterrows():
    if row['Target'] not in targets:
        targets.append(row['Target'])

for t in targets:
    task4(train, test, t)


