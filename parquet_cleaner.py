import pandas as pd

df = pd.read_parquet('train.parquet')

# Удаление строк, где 'shell' — NaN или пустая строка
df_cleaned = df[df['shell'].notna() & (df['shell'].str.strip() != '')]

df_cleaned.to_parquet('train_cleaned.parquet')