from sklearn.model_selection import train_test_split
import pandas as pd
path = 'C:/Users/Pranjali/cleaned_data.csv'
df = pd.read_csv(path)

target = ' default payment next month'
features = ['LIMIT_BAL','EDUCATION','MARRIAGE','AGE','PAY_1','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
            'PAY_AMT5', 'PAY_AMT6']

Y = df['default payment next month']
X = df[features]

x_train,x_test,y_train,y_test= train_test_split(X, Y)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

import pickle
pickle.dump(clf,open('credit_card_model.pkl','wb'))
