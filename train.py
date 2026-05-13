import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_excel("electricity_dataset_5020.xlsx")

df = df.drop(['Meter_No','House_ID'], axis=1, errors='ignore')
df['Last_Payment_Date'] = df['Last_Payment_Date'].fillna("Not Paid")
df = df.fillna(df.median(numeric_only=True))

df = pd.get_dummies(df, drop_first=True)

X = df.drop([
    'Monthly_Savings',
    'Solar_kWh_month',
    'Monthly_Bill',
    'Yearly_Bill'
], axis=1)

y = df['Monthly_Savings']

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))