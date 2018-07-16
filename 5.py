import pandas as pd

fields = ['type', 'word', 'pos', 'priorpolarity']
lex = pd.read_csv('/Users/krishna/Documents/8_CLA/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.csv', engine='python', usecols=fields, delim_whitespace=True)

dfl = pd.DataFrame(lex)

dfl['type'] = dfl['type'].map(lambda x: str(x)[5:])
dfl['word'] = dfl['word'].map(lambda x: str(x)[6:])
dfl['pos'] = dfl['pos'].map(lambda x: str(x)[5:])
dfl['priorpolarity'] = dfl['priorpolarity'].map(lambda x: str(x)[14:])

#print(dfl)

for index, row in dfl.iterrows():
    if row['priorpolarity'] == 'positive':
        row['priorpolarity'] = 1
    elif row['priorpolarity'] == 'negative':
        row['priorpolarity'] = -1
    else:
        row['priorpolarity'] = 0
    if row['type'] == 'weaksubj':
        row['type'] = 1
    elif row['type'] == 'strongsubj':
        row['type'] = 2

dfl['score'] = dfl['type']*dfl['priorpolarity']

#print(dfl)

from stanza.nlp.corenlp import CoreNLPClient
from sklearn.feature_extraction.text import CountVectorizer

client = CoreNLPClient(server='http://localhost:9000', default_annotators=['ssplit', 'tokenize', 'pos'])

fields = ['Tweet', 'Target', 'Stance']
train = pd.read_csv('/Users/krishna/Desktop/PycharmProjects/cla5/train.csv', engine='python', usecols=fields)
test = pd.read_csv('/Users/krishna/Desktop/PycharmProjects/cla5/test.csv', engine='python', usecols=fields)

def task1(train_data, test_data, target):

    print(target)

    twts =[]
    s = []
    j = " "

    for index, row in train_data.iterrows():
        if row['Target'] == target:
            line = row['Tweet']
            split = line.split()
            for string in split:
                if string.startswith('#') | string.startswith('@'):
                    s.append(string)
            new_split = [string for string in split if string not in s]
            tweet = j.join(new_split)
            twts.append(tweet)

    annotated = client.annotate(str(twts))

    bag_of_words = []

    for sentence in annotated.sentences:
        for token in sentence:
            if token.pos.startswith('N') | token.pos.startswith('V') | token.pos.startswith('J') :
                bag_of_words.append(token.word)

    #print(len(bag_of_words))

    features = list(set(bag_of_words))

    #print(features)
    #print(len(features))

    cv = CountVectorizer(vocabulary=features)

    train_tweets = []
    train_stance = []

    for index, row in train_data.iterrows():
        if row['Target'] == target:
            train_tweets.append(row['Tweet'])
            train_stance.append(row['Stance'])

    traintweets_df = pd.DataFrame({'Tweet':train_tweets})
    trainstance_df = pd.DataFrame({'Stance':train_stance})

    train_vectors = cv.fit_transform(traintweets_df['Tweet'])
    train_vectors1 = train_vectors.toarray()

    df1 = pd.DataFrame(train_vectors1, columns=cv.get_feature_names())
    #print(df1.head())

    df2 = pd.concat([df1, trainstance_df], axis=1)

    #print(df2)

    s = df2.eq(1).stack()
    s.append(df2.eq(2).stack())
    s.append(df2.eq(3).stack())
    s.append(df2.eq(4).stack())
    s.append(df2.eq(5).stack())

    location = s[s].index.values

    #print(location)

    for x,y in location:
        ly = y.lower()
        for w in dfl['word']:
            if ly == w:
                #print(ly)
                i = dfl.loc[dfl['word'] == ly].index[0]
                df2.loc[x,y] = dfl.loc[i, 'score']

    print(df2)

    output = 'ltrain_'+target[:3]+'.csv'

    #df2.to_csv(output, index=False, header=None, encoding='utf-8')

    test_tweets = []
    test_stance = []

    for index, row in test_data.iterrows():
        if row['Target'] == target:
            test_tweets.append(row['Tweet'])
            test_stance.append(row['Stance'])

    testtweets_df = pd.DataFrame({'Tweet':test_tweets})
    teststance_df = pd.DataFrame({'Stance':test_stance})

    test_vectors = cv.fit_transform(testtweets_df['Tweet'])
    test_vectors1 = test_vectors.toarray()

    df3 = pd.DataFrame(test_vectors1, columns=cv.get_feature_names())
    #print(df3.head())

    df4 = pd.concat([df3, teststance_df], axis=1)

    #print(df4)

    k = df4.eq(1).stack()
    locat = k[k].index.values
    #print(locat)

    for x1,y1 in locat:
        ly1 = y1.lower()
        for w1 in dfl['word']:
            if ly1 == w1:
                #print(ly1)
                i1 = dfl.loc[dfl['word'] == ly1].index[0]
                df4.loc[x1,y1] = dfl.loc[i1, 'score']

    print(df4)

    output = 'ltest_'+target[:3]+'.csv'

    #df4.to_csv(output, index=False, header=None, encoding='utf-8')

    return 1

targets = []

for index, row in train.iterrows():
    if row['Target'] not in targets:
        targets.append(row['Target'])

for t in targets:
    task1(train, test, t)

#task1(train, test, 'Atheism')
