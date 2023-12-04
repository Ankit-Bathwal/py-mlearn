import pandas as pd 

# Get data
df = pd.read_csv('https://raw.githubusercontent.com/Ankit-Bathwal/py-mlearn/main/firstproj/avgIQpercountry.csv')

# Data prep
y = df['Average IQ']
print(repr(y))

X = df.drop('Average IQ',axis=1)
print(repr(X))

# Data splitting
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=0.2)
 
# Model building - linear regression
from sklearn.linear_model import LinearRegression 
lr = LinearRegression()
lr.fit(X_train, y_train)

y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

print(repr(y_lr_train_pred))
