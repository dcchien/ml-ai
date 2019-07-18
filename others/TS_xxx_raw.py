#
import datetime as datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/xxx_Raw_Data-0827-0831.csv', header = 0, parse_dates=True)
# 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#print(df.head(10))
#print(df.dtypes)
print("Raw Data/shape =", df.shape)

# drop the following columns
cols = ['BUS_PROCESS_GRP', 'VALUATION_DATE', 'END_TIME', 'PARTITION_ID', 'TIME']

# inplace=False
#df = df.drop(cols, axis=1)
# inplace=True
df.drop(cols, axis=1, inplace=True)
df.rename(columns=str.lower, inplace=True)
df.rename(
	columns={'statistics_name': 'stats_name',
			 'time_taken_in_seconds': 'taken_sec',
			 'user_field3': 'user'
			},
	inplace=True
)

#df['taken_sec'] = df['time_taken_in_seconds']

df['start_time'] = pd.to_datetime(df['start_time'])
#write to a csv file
df.to_csv('./data/TS_xxx_data2.csv', encoding='utf-8', index=False)

print(df.head(20))
print(df.tail(20))
print(df.shape)
print(df.dtypes)
print(df.info())
print(df.describe())

# graph
sns.relplot(x='stats_name', y='taken_sec', data=df)
plt.show()

sns.distplot(df[taken_sec'], bins=10)
plt.show()

#groupby multiple cols
df = df.groupby(['stats_name', 'entity_id'])

print(df.head())
print(df.describe())

#sns.hist(df, hue='taken_sec')


