
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
X = df.iloc[:,0:2]
print(X)
y = df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter = 1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy:{accuracy}')

import joblib
joblib.dump(model,'model.pkl')

loaded_model = joblib.load('model.pkl') 
prediction = loaded_model.predict(X_test)