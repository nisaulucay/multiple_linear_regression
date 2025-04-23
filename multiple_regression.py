import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import joblib 

# veriyi oku
df = pd.read_csv("multipledata.csv")

# özellik ve hedef
X = df[['deneyim','yas']]
y = df[['maas']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model,'multiple_linear_model.pkl') 

#tahmin
print("15 yıl deneyim ve 40 yaş için için maaş tahmini:", model.predict([[15,40]]))