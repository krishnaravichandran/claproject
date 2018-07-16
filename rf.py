import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = [('pos_train_Ath.csv','pos_test_Ath.csv'),('pos_train_Cli.csv','pos_test_Cli.csv'),('pos_train_Fem.csv','pos_test_Fem.csv'),('pos_train_Hil.csv','pos_test_Hil.csv'),('pos_train_Leg.csv','pos_test_Leg.csv'),
        ('all_train_Ath.csv','all_test_Ath.csv'),('all_train_Cli.csv','all_test_Cli.csv'),('all_train_Fem.csv','all_test_Fem.csv'),('all_train_Hil.csv','all_test_Hil.csv'),('all_train_Leg.csv','all_test_Leg.csv'),
        ('lex_train_Ath.csv','lex_test_Ath.csv'),('lex_train_Cli.csv','lex_test_Cli.csv'),('lex_train_Fem.csv','lex_test_Fem.csv'),('lex_train_Hil.csv','lex_test_Hil.csv'),('lex_train_Leg.csv','lex_test_Leg.csv'),
        ('ntrain_Ath.csv','ntest_Ath.csv'),('ntrain_Cli.csv','ntest_Cli.csv'),('ntrain_Fem.csv','ntest_Fem.csv'),('ntrain_Hil.csv','ntest_Hil.csv'),('ntrain_Leg.csv','ntest_Leg.csv'),
        ('ath_triple.csv','ath_t_triple.csv'),('cli_triple.csv','cli_t_triple.csv'),('fem_triple.csv','fem_t_triple.csv'),('hil_triple.csv','hil_t_triple.csv'),('leg_triple.csv','leg_t_triple.csv'),
        ('ath_chain.csv','ath_t_chain.csv'),('cli_chain.csv','cli_t_chain.csv'),('fem_chain.csv','fem_t_chain.csv'),('hil_chain.csv','hil_t_chain.csv'),('leg_chain.csv','leg_t_chain.csv')]


def rf(i1, i2):

    #print(i2)

    train = pd.read_csv('/Users/krishna/Desktop/rf/'+i1, engine='python')
    test = pd.read_csv('/Users/krishna/Desktop/rf/'+i2, engine='python')

    train_features = train.iloc[:-1,:-1]
    train_target = train.iloc[:-1,-1]

    test_features = test.iloc[:-1,:-1]
    test_target = test.iloc[:-1,-1]

    #print(train_features.head())
    #print(train_target.head())

    clf = RandomForestClassifier()
    clf = clf.fit(train_features, train_target)

    pred_target = clf.predict(test_features)

    print((pred_target))

    return 1

for (i1, i2) in data:
    rf(i1, i2)



