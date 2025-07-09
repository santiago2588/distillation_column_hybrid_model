import pandas as pd

file_path = '/home/ubuntu/upload/DistillationColumnDataset.xlsx'
df = pd.read_excel(file_path)

print('Dataset Head:')
print(df.head())

print('\nDataset Info:')
df.info()

print('\nDataset Description:')
print(df.describe())


