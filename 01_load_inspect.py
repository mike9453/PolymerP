import pandas as pd

df = pd.read_csv('train.csv')
print("總筆數與欄位數：", df.shape)
print("欄位名稱：", df.columns.tolist())
print("\n各屬性非空數量：")
print(df[['Tg','FFV','Tc','Density','Rg']].notnull().sum())
print("\n前 5 筆：")
print(df.head())
