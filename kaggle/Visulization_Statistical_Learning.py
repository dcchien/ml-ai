# Kaggle - Statistical Learning for beginner
# Visulization
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/downloads/data.csv/notebook
# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from pandas.tools import plotting
from scipy import stats
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../data"))

#load and read the dataset as pandas dataframe
df = pd.read_csv("../data/data_wis_breast_cancer.csv")
df = df.drop(['Unnamed: 32','id'], axis = 1)

# quick look of dataset
print(df.head())
print(df.shape)
print(df.columns)

#
# plot
#
#How many times each value appears in dataset. This description is called the distribution of variable
#Most common way to represent distribution of varible is histogram that is graph which shows frequency of each value.
#Frequency = number of times each value appears
# Example: [1,1,1,1,2,2,2]. Frequency of 1 is four and frequency of 2 is three.
# Higtogram
m = plt.hist(df[df["diagnosis"] == "M"].radius_mean,bins=30,fc = (1,0,0,0.5),label = "Malignant")
b = plt.hist(df[df["diagnosis"] == "B"].radius_mean,bins=30,fc = (0,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors")
plt.show()
frequent_malignant_radius_mean = m[0].max()
index_frequent_malignant_radius_mean = list(m[0]).index(frequent_malignant_radius_mean)
most_frequent_malignant_radius_mean = m[1][index_frequent_malignant_radius_mean]
print("Most frequent malignant radius mean is: ",most_frequent_malignant_radius_mean)

# Outlier
#Calculating outliers:
#first we need to calculate first quartile (Q1)(25%)
#    then find IQR(inter quartile range) = Q3-Q1
#    finally compute Q1 - 1.5IQR and Q3 + 1.5IQR
#    Anything outside this range is an outlier
#    lets write the code for bening tumor distribution for feature radius mean

data_bening = df[df["diagnosis"] == "B"]
data_malignant = df[df["diagnosis"] == "M"]
desc = data_bening.radius_mean.describe()
Q1 = desc[4]
Q3 = desc[6]
IQR = Q3-Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
print("Anything outside this range is an outlier: (", lower_bound ,",", upper_bound,")")
data_bening[data_bening.radius_mean < lower_bound].radius_mean
print("Outliers: ",data_bening[(data_bening.radius_mean < lower_bound) | (data_bening.radius_mean > upper_bound)].radius_mean.values)

# Box Plot
melted_data = pd.melt(df,id_vars = "diagnosis",value_vars = ['radius_mean', 'texture_mean'])
plt.figure(figsize = (8,6))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()

# We found 3 outlier in bening radius mean and in box plot there are 3 outlier.

# Summary statistics
print("mean: ",data_bening.radius_mean.mean())
print("variance: ",data_bening.radius_mean.var())
print("standart deviation (std): ",data_bening.radius_mean.std())
print("describe method: ",data_bening.radius_mean.describe())

#CDF Cumulative Distribution FUnction

#Cumulative distribution function is the probability that the variable takes a value less than or equal to x. P(X <= x)
#Lets explain in cdf graph of bening radiues mean
#in graph, what is P(12 < X)? The answer is 0.5. The probability that the variable takes a values less than or equal to 12(radius mean) is 0.5.
#You can plot cdf with two different method

plt.hist(data_bening.radius_mean,bins=50,fc=(0,1,0,0.5),label='Bening',normed = True,cumulative = True)
sorted_data = np.sort(data_bening.radius_mean)
y = np.arange(len(sorted_data))/float(len(sorted_data)-1)
plt.plot(sorted_data,y,color='red')
plt.title('CDF of bening tumor radius mean')
plt.show()

# Effective size
mean_diff = data_malignant.radius_mean.mean() - data_bening.radius_mean.mean()
var_bening = data_bening.radius_mean.var()
var_malignant = data_malignant.radius_mean.var()
var_pooled = (len(data_bening)*var_bening +len(data_malignant)*var_malignant ) / float(len(data_bening)+ len(data_malignant))
effect_size = mean_diff/np.sqrt(var_pooled)
print("Effect size: ",effect_size)

# Relationship between variables
#
# We can say that two variables are related with each other, if one of them gives information about others
#    For example, price and distance. If you go long distance with taxi you will pay more. There fore we can say that price and distance are positively related with each other.
#    Scatter Plot
#    Simplest way to check relationship between two variables
#    Lets look at relationship between radius mean and area mean
#    In scatter plot you can see that when radius mean increases, area mean also increases. Therefore, they are positively correlated with each other.
#    There is no correlation between area mean and fractal dimension se. Because when area mean changes, fractal dimension se is not affected by chance of area mean

plt.figure(figsize = (10,6))
sns.jointplot(df.radius_mean,df.area_mean,kind="regg")
plt.show()

# Also we can look relationship between more than 2 distribution

sns.set(style = "white")
df2 = df.loc[:,["radius_mean","area_mean","fractal_dimension_se"]]
g = sns.PairGrid(df2,diag_sharey = False,)
g.map_lower(sns.kdeplot,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,lw =3)
plt.show()

# Correlation

f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(df.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('../graphs/graph.png')
plt.show()

# Covariance

#    Covariance is measure of the tendency of two variables to vary together
#    So covariance is maximized if two vectors are identical
#    Covariance is zero if they are orthogonal.
#    Covariance is negative if they point in opposite direction
#    Lets look at covariance between radius mean and area mean. Then look at radius mean and fractal dimension se

np.cov(df.radius_mean,df.area_mean)
print("Covariance between radius mean and area mean: ",df.radius_mean.cov(df.area_mean))
print("Covariance between radius mean and fractal dimension se: ",df.radius_mean.cov(df.fractal_dimension_se))

# Perason Correlation
p1 = df.loc[:,["area_mean","radius_mean"]].corr(method= "pearson")
p2 = df.radius_mean.cov(df.area_mean)/(df.radius_mean.std()*df.area_mean.std())
print('Pearson correlation: ')
print(p1)
print('Pearson correlation: ',p2)

#Spearman's Rank Correlation

ranked_data = df.rank()
spearman_corr = ranked_data.loc[:,["area_mean","radius_mean"]].corr(method= "pearson")
print("Spearman's correlation: ")
print(spearman_corr)

# Spearman's correlation is little higher than pearson correlation
#If relationship between distributions are non linear, spearman's correlation tends to better estimate the strength of relationship
#Pearson correlation can be affected by outliers. Spearman's correlation is more robust.

# Hypothesis Testing
statistic, p_value = stats.ttest_rel(df.radius_mean, df.area_mean)
print('p-value: ',p_value)

#P values is almost zero so we can reject null hypothesis.

#Normal(Gaussian) Distribution and z-score
# parameters of normal distribution

mu, sigma = 110, 20  # mean and standard deviation
s = np.random.normal(mu, sigma, 100000)
print("mean: ", np.mean(s))
print("standart deviation: ", np.std(s))
# visualize with histogram
plt.figure(figsize = (10,7))
plt.hist(s, 100, normed=False)
plt.ylabel("frequency")
plt.xlabel("IQ")
plt.title("Histogram of IQ")
plt.show()

