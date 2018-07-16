import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

fields = ['ID', 'Word', 'Head', 'Label']

train = pd.read_csv('/Users/krishna/Desktop/PycharmProjects/claproject/Leg_train_out.conll', sep='\t', engine='python', usecols=fields)
test = pd.read_csv('/Users/krishna/Desktop/PycharmProjects/claproject/Leg_test_out.conll', sep='\t', engine='python', usecols=fields)

df = pd.DataFrame(train)

#print(train)

#print(test)

ids = []


j = 1
start = 0
end = 0

for i in range(0, len(train)):
    #print(row['ID'])
    if j == train.iloc[i]['ID']:
        #print(train.iloc[i]['ID'])
        j = j + 1
        end = i
    else:
        #print("end")
        ids.append((start,end+1))
        #print(train.iloc[i]['ID'])
        j = 2
        start = i

#print(ids)

triplefeatures = []
chainfeatures = []

triplesentences = []
chainsentences = []

for (a,b) in ids:
    train['triple'] = train.iloc[1]['Word']+train.iloc[train.iloc[1,2]-1,1]+train.iloc[1]['Label']
    train['chain'] = train.iloc[1]['Word']+train.iloc[train.iloc[1,2]-1,1]+train.iloc[1]['Label']
    j1 = ''
    j2 = ''
    for k in range(a, b):
        #print(train.iloc[k,2])
        j1 = j1 + " " + str(train.iloc[k]['Word']) + str(train.iloc[train.iloc[k,2]-1+a,1]) + str(train.iloc[k]['Label']);
        train.at[k,'triple'] = str(train.iloc[k]['Word']) + str(train.iloc[train.iloc[k,2]-1+a,1]) + str(train.iloc[k]['Label']);
        triplefeatures.append(train.at[k,'triple'])
        temp = ""
        count = 0
        k_temp = k
        temp = temp + str(train.iloc[k_temp]['Word'])
        while True:
            if count == 3:
                break
            if train.iloc[k_temp,2] == 0:
                temp = temp + str(train.iloc[train.iloc[k_temp,2]-1+a,1])
                break

            temp = temp + str(train.iloc[train.iloc[k_temp,2]-1+a,1])
            count = count + 1
            k_temp = train.iloc[k_temp,2]-1+a
        j2 = j2 + " "+ temp;
        train.at[k,'chain'] = temp
        chainfeatures.append(train.at[k,'chain'])
    triplesentences.append(j1);
    chainsentences.append(j2);
    #print(train.iloc[a:b, 0:6])

print(len(triplefeatures))
print(len(chainfeatures))

#print(triplesentences)
#print(chainsentences)




tdf = pd.DataFrame(test)

#print(train)

#print(test)

tids = []


tj = 1
tstart = 0
tend = 0

for i in range(0, len(test)):
    #print(row['ID'])
    if tj == test.iloc[i]['ID']:
        #print(train.iloc[i]['ID'])
        tj = tj + 1
        tend = i
    else:
        #print("end")
        tids.append((tstart,tend+1))
        #print(train.iloc[i]['ID'])
        tj = 2
        tstart = i

#print(ids)

t_triplefeatures = []
t_chainfeatures = []

t_triplesentences = []
t_chainsentences = []

for (a,b) in tids:
    test['triple'] = test.iloc[1]['Word']+test.iloc[test.iloc[1,2]-1,1]+test.iloc[1]['Label']
    test['chain'] = test.iloc[1]['Word']+test.iloc[test.iloc[1,2]-1,1]+test.iloc[1]['Label']
    j1 = ''
    j2 = ''
    for k in range(a, b):
        #print(train.iloc[k,2])
        j1 = j1 + " " + str(test.iloc[k]['Word']) + str(test.iloc[test.iloc[k,2]-1+a,1]) + str(test.iloc[k]['Label']);
        test.at[k,'triple'] = str(test.iloc[k]['Word']) + str(test.iloc[test.iloc[k,2]-1+a,1]) + str(test.iloc[k]['Label']);
        t_triplefeatures.append(test.at[k,'triple'])
        temp = ""
        count = 0
        k_temp = k
        temp = temp + str(test.iloc[k_temp]['Word'])
        while True:
            if count == 3:
                break
            if test.iloc[k_temp,2] == 0:
                temp = temp + str(test.iloc[test.iloc[k_temp,2]-1+a,1])
                break

            temp = temp + str(test.iloc[test.iloc[k_temp,2]-1+a,1])
            count = count + 1
            k_temp = test.iloc[k_temp,2]-1+a
        j2 = j2 + " "+ temp;
        test.at[k,'chain'] = temp
        t_chainfeatures.append(test.at[k,'chain'])
    t_triplesentences.append(j1);
    t_chainsentences.append(j2);
    #print(train.iloc[a:b, 0:6])

#print(t_triplefeatures)
#print(chainfeatures)

#print(t_triplesentences)
#print(t_chainsentences)



df1 = pd.DataFrame(t_triplesentences)

features1 = list(set(triplefeatures))

cv1 = CountVectorizer(vocabulary=features1)

vectors = cv1.fit_transform(df1[0])
vectors1 = vectors.toarray()

df2 = pd.DataFrame(vectors1, columns=cv1.get_feature_names())
#print(df2)

#df2.to_csv('leg_t_triple.csv', index=False, header=None, encoding='utf-8')


df3 = pd.DataFrame(t_chainsentences)

features2 = list(set(chainfeatures))

cv2 = CountVectorizer(vocabulary=features2)

cvectors = cv2.fit_transform(df3[0])
cvectors1 = cvectors.toarray()

df4 = pd.DataFrame(cvectors1, columns=cv2.get_feature_names())
#print(df4)

#df4.to_csv('leg_t_chain.csv', index=False, header=None, encoding='utf-8')

