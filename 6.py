import pandas as pd
from stanza.nlp.corenlp import CoreNLPClient
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
import re

client = CoreNLPClient(server='http://localhost:9000', default_annotators=['ssplit', 'tokenize', 'pos'])

fields = ['Tweet', 'Target', 'Stance']

train = pd.read_csv('/Users/krishna/Desktop/PycharmProjects/cla5/train.csv', engine='python', usecols=fields)
test = pd.read_csv('/Users/krishna/Desktop/PycharmProjects/cla5/test.csv', engine='python', usecols=fields)


def task6(train_data, test_data, target):

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

    train_tweets = pd.DataFrame({'Tweet':traintwts})

    #print(train_tweets)

    annotated = client.annotate(str(traintwts))

    bag_of_words = []

    for sentence in annotated.sentences:
        for token in sentence:
            if token.pos.startswith('N') | token.pos.startswith('V') | token.pos.startswith('J') :
                bag_of_words.append(token.word)

    #print(len(bag_of_words))

    features1 = list(set(bag_of_words))
    print(len(features1))

    cv1 = CountVectorizer(vocabulary=features1)

    train_vectors = cv1.fit_transform(train_tweets['Tweet'])
    train_vectors1 = train_vectors.toarray()

    df1 = pd.DataFrame(train_vectors1, columns=cv1.get_feature_names())
    #print(df1.head())


    ntraintwts = []
    ntrainj = ''

    for index, row in train_tweets.iterrows():
        l = row['Tweet']
        l1 = re.sub('\s+', '_', l)
        #print(l1)
        grams = ngrams(list(l1),8)
        gsent = ""
        for gram in grams:
            jgrams = ntrainj.join(gram)
            gsent = gsent+" "+jgrams
        #print(gsent)
        ntraintwts.append(gsent)

    #print(ntraintwts)

    ntrain_tweets = pd.DataFrame({'nTweet':ntraintwts})

    #print(ntrain_tweets)

    n_grams = []

    for sent in ntraintwts:
        words = sent.split()
        n_grams.append(words)

    #print(n_grams)

    flat_list = []

    for ng in n_grams:
        for g in ng:
            flat_list.append(g)

    features2 = list(set(flat_list))
    print(len(features2))

    print(len(features1)+len(features2))

    cv2 = CountVectorizer(vocabulary=features2)

    ntrain_vectors = cv2.fit_transform(ntrain_tweets['nTweet'])
    ntrain_vectors1 = ntrain_vectors.toarray()

    ndf1 = pd.DataFrame(ntrain_vectors1, columns=cv2.get_feature_names())
    #print(ndf1.head())


    train_stance = []

    for index, row in train_data.iterrows():
        if row['Target'] == target:
            train_stance.append(row['Stance'])

    trainstance_df = pd.DataFrame({'Stance':train_stance})

    df2 = pd.concat([df1, ndf1, trainstance_df], axis=1)

    #print(df2)

    output = 'ntrain_'+target[:3]+'.csv'

    #df2.to_csv(output, index=False, header=None, encoding='utf-8')



    #print(target)

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

    test_tweets = pd.DataFrame({'Tweet':testtwts})

    #print(test_tweets)

    test_vectors = cv1.fit_transform(test_tweets['Tweet'])
    test_vectors1 = test_vectors.toarray()

    df3 = pd.DataFrame(test_vectors1, columns=cv1.get_feature_names())
    #print(df3)


    ntesttwts = []
    ntestj = ''

    for index, row in test_tweets.iterrows():
        l = row['Tweet']
        l1 = re.sub('\s+', '_', l)
        #print(l1)
        grams = ngrams(list(l1),8)
        gsent = ""
        for gram in grams:
            jgrams = ntestj.join(gram)
            gsent = gsent+" "+jgrams
        #print(gsent)
        ntesttwts.append(gsent)

    #print(ntwts)

    ntest_tweets = pd.DataFrame({'nTweet':ntesttwts})

    #print(ntest_tweets)

    ntest_vectors = cv2.fit_transform(ntest_tweets['nTweet'])
    ntest_vectors1 = ntest_vectors.toarray()

    ndf3 = pd.DataFrame(ntest_vectors1, columns=cv2.get_feature_names())
    #print(ndf3)

    test_stance = []

    for index, row in test_data.iterrows():
        if row['Target'] == target:
            test_stance.append(row['Stance'])

    teststance_df = pd.DataFrame({'Stance':test_stance})

    df4 = pd.concat([df3, ndf3, teststance_df], axis=1)

    #print(df4)

    output = 'ntest_'+target[:3]+'.csv'

    #df4.to_csv(output, index=False, header=None, encoding='utf-8')

    return 1

targets = []

for index, row in train.iterrows():
    if row['Target'] not in targets:
        targets.append(row['Target'])

for t in targets:
    task6(train, test, t)
