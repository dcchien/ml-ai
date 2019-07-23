# 
import datetime as datetime
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
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
# df = df.drop(cols, axis=1)
# or inplace=True
df.drop(cols, axis=1, inplace=True)
df.rename(columns=str.lower, inplace=True)
df.rename(
	columns={'statistics_name': 'stats_name',
			 'time_taken_in_seconds': 'taken_sec',
			 'user_field3': 'user'
			},
	inplace=True
)

# split objects of data and time into two columns
df['sdate'] = df['start_time'].str.split(' ').str[0]
df['stime'] = df['start_time'].str.split(' ').str[1]
df.drop('start_time', axis=1, inplace=True)

# shuffle colns
cols = ['sdate', 'stime', 'stats_name', 'entity_id', 'user', 'taken_sec']
df = df[cols]
print(df.head())
# check missing data
print("NaN Counts")
print(df.isna().sum())

df1 = df.groupby('user')['taken_sec'].count()
print(df1.head())
#write to a csv file
df.to_csv('./data/TS_xxx_data2.csv', encoding='utf-8', index=False)

print(df.shape)
print(df.dtypes)
print(df.info())
#print(df.describe())

# Graphs 
sns.relplot(x='stats_name', y='taken_sec', hue='stats_name', data=df)
plt.xticks(rotation=45, fontsize=5)
plt.show()

sns.distplot(df['taken_sec'], kde=True, bins=10)
plt.show()

sns.boxplot(x=df['stats_name'], y=df['taken_sec'], width=0.5)
plt.xticks(rotation=45, fontsize=5)
plt.show()

#groupby multiple cols
#df = df.groupby(['stats_name', 'entity_id'])

print(df.head())
print(df.describe())

