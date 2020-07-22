---
title: Corporate Credit Rating Forecasting
image: /assets/images/finance.jpg
author: Alan Gewerc
date: November 2018
categories:
  - Data Science
  - Finance
  - Machine Learning
  - Deep learning
layout: defaultpost
---


### Introduction

This notebook contains the results of the data analysis performed on a set of corporate credit ratings given by ratings agencies to a set of companies. The aim of the data analysis is to build a machine learning model from the rating data that can be used to predict the rating a company will receive.

The first section section of the notebook shows the exploratory data analysis (EDA) performed to explore and understand the data. It looks at each attribute (variable) in the data to understand the nature and distribution of the attribute values. It also examines the correlation between the variables through visual analysis. A summary at the end highlights the key findings of the EDA.

The second section shows the development of a machine learning model. Many diffferent models are tested and the performance of all models are compared. Subsequently, a winner is selected and we do hyperparameter tunning.

In the model evaluation step we use different techniques such as a confusion matrix and scores as F1, Precision and Recall to understand different aspects of the performance of the model. We also perform feature selection to know what financial indicators are more relevant for the rating agencies. 


### The Dataset

There are 30 features for every company of which 25 are financial indicators. 
They can be divided in: <br>
- **Liquidity Measurement Ratios**: currentRatio, quickRatio, cashRatio, daysOfSalesOutstanding
- **Profitability Indicator Ratios**: grossProfitMargin, operatingProfitMargin, pretaxProfitMargin, netProfitMargin, effectiveTaxRate, returnOnAssets, returnOnEquity, returnOnCapitalEmployed
- **Debt Ratios**: debtRatio, debtEquityRatio
- **Operating Performance Ratios**: assetTurnover
- **Cash Flow Indicator Ratios**: operatingCashFlowPerShare, freeCashFlowPerShare, cashPerShare, operatingCashFlowSalesRatio, freeCashFlowOperatingCashFlowRatio 


### Libraries used:
- pandas
- numpy
- matplotlib
- seaborn
- random
- sklearn
- xgboost
- wordcloud

Load the Libraries used in the notebook

```
import pandas as pdf
import numpy as np
from numpy import loadtxt
from numpy import sort
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as mtick
from wordcloud import WordCloud, STOPWORDS 
from random import sample
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn  ....
```

<br>
#### Import Dataset

```python
df_rating = pd.read_csv('data/corporate_rating.csv')
```

## Exploratory Data Analysis

Our first step is to perform an exploratory data analysis to understand the charateristics of dataset. Here are some quesitons we will try to adress:

- What are the dimensions of the data?
- How do predictors relate to each other?
- What are the classes of the data?
- How are the predictors distributed?
- How are the labels distributed?
- Do we have missing values?
- Are outliers are relevant?
- Are there any transformations that must be done with the dataset?


```python
# Display the dimensions
print("The credit rating dataset has", df_rating.shape[0], "records, each with", df_rating.shape[1],
    "attributes")
```
<samp>
The credit rating dataset has 2029 records, each with 31 attributes
</samp>    

We will now use the function `.info()` to see the classes of columns and search missing values.  


```python
# Display the structure
df_rating.info()
```

<samp class="text-primary">
RangeIndex: 2029 entries, 0 to 2028<br>
Data columns (total 31 columns):<br>
Rating                                2029 non-null object<br>
Name                                  2029 non-null object<br>
Symbol                                2029 non-null object<br>
Rating Agency Name                    2029 non-null object<br>
Date                                  2029 non-null object<br>
Sector                                2029 non-null object<br>
currentRatio                          2029 non-null float64<br>
quickRatio                            2029 non-null float64<br>
cashRatio                             2029 non-null float64<br>
daysOfSalesOutstanding                2029 non-null float64<br>
netProfitMargin                       2029 non-null float64<br>
pretaxProfitMargin                    2029 non-null float64<br>
grossProfitMargin                     2029 non-null float64<br>
operatingProfitMargin                 2029 non-null float64<br>
returnOnAssets                        2029 non-null float64<br>
...
dtypes: float64(25), object(6)<br>
memory usage: 491.5+ KB<br>
</samp>

<br>
We have 26 columns of numerical data and 6 descriptive columns (one of which is the label).There are no missing values.
<br> A first look at the data:
<br>

```python
df_rating.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rating</th>
      <th>Name</th>
      <th>Symbol</th>
      <th>Rating Agency Name</th>
      <th>Date</th>
      <th>Sector</th>
      <th>currentRatio</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>A</td>
      <td>Whirlpool Corporation</td>
      <td>WHR</td>
      <td>Egan-Jones Ratings Company</td>
      <td>11/27/2015</td>
      <td>Consumer Durables</td>
      <td>0.945894</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>BBB</td>
      <td>Whirlpool Corporation</td>
      <td>WHR</td>
      <td>Egan-Jones Ratings Company</td>
      <td>2/13/2014</td>
      <td>Consumer Durables</td>
      <td>1.033559</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>BBB</td>
      <td>Whirlpool Corporation</td>
      <td>WHR</td>
      <td>Fitch Ratings</td>
      <td>3/6/2015</td>
      <td>Consumer Durables</td>
      <td>0.963703</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>BBB</td>
      <td>Whirlpool Corporation</td>
      <td>WHR</td>
      <td>Fitch Ratings</td>
      <td>6/15/2012</td>
      <td>Consumer Durables</td>
      <td>1.019851</td>
      <td>...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>BBB</td>
      <td>Whirlpool Corporation</td>
      <td>WHR</td>
      <td>Standard &amp; Poor's Ratings Services</td>
      <td>10/24/2016</td>
      <td>Consumer Durables</td>
      <td>0.957844</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



### Analyse Labels

As we know we are working with ordinal labels. That means there is a scale from more secure to less secure ratings. For instance, the triple-A (AAA) is the most secure rating a company can receive. On the other hand, the rating D is the less secure. It means the company will likely default on its creditors. Let's have a first look at the how many reatings we have of each in the dataset. 



```python
df_rating.Rating.value_counts()
```

<samp class="text-primary">
    BBB    671<br>
    BB     490<br>
    A      398<br>
    B      302<br>
    AA      89<br>
    CCC     64<br>
    AAA      7<br>
    CC       5<br>
    C        2<br>
    D        1<br>
    Name: Rating, dtype: int64<br>
</samp>
<br>

We observe that the dataset is very unbalanced. We have 671 triple-Bs (BBB) but only 1 D. However, we are working with Ratings from different companies such as `Moody's`, `Standard & Poor's` and more. Therefore it is preferred to simplify the labels according to this table from the website [investopedia](https://www.investopedia.com/terms/c/corporate-credit-rating.asp). We will classify our labels according to the grading risk and not the rate. 

| Bond Rating                                                            |
|-------------|-------------------|----------|------------|--------------|
| Moody's     | Standard & Poor's | Fitch    | Grade      | Risk         |
|-------------|-------------------|----------|------------|--------------|
| Aaa         | AAA               | AAA      | Investment | Lowest Risk  |
| Aa          | AA                | AA       | Investment | Low Risk     |
| A           | A                 | A        | Investment | Low Risk     |
| Baa         | BBB               | BBB      | Investment | Medium Risk  |
| Ba, B       | BB, B             | BB, B    | Junk       | High Risk    |
| Caa/Ca      | CCC/CC/C          | CCC/CC/C | Junk       | Highest Risk |
| C           | D                 | D        | Junk       | In Default   |


To do it we will replace with a dictonary each of this ratings. 


```python
rating_dict = {'AAA':'Lowest Risk', 
               'AA':'Low Risk',
               'A':'Low Risk',
               'BBB':'Medium Risk', 
               'BB':'High Risk',
               'B':'High Risk',
               'CCC':'Highest Risk', 
               'CC':'Highest Risk',
               'C':'Highest Risk',
               'D':'In Default'}

df_rating.Rating = df_rating.Rating.map(rating_dict)
```


```python
ax = df_rating['Rating'].value_counts().plot(kind='bar',
                                             figsize=(8,4),
                                             title="Count of Rating by Type",
                                             grid=True)
```


![png](/assets/images/corporateCreditRatingPrediction-Copy1_17_0.png)


Unfortunately, given the lack of Credit Ratings classified as `Lowest Risk` and `In Default` we will have to eliminate then from the table. However, the dataset will keep unbalanced and if needed we will have to adress this issue in further steps. 


```python
df_rating = df_rating[df_rating['Rating']!='Lowest Risk'] # filter Lowest Risk
df_rating = df_rating[df_rating['Rating']!='In Default']  # filter In Default
df_rating.reset_index(inplace = True, drop=True) # reset index
```

### Descriptive Statistics

Now we will use statistical tools, especially from pandas to improve the understanding from the dataset, especially the numerical features. We have seen there are 25 numerical columns in the dataset, all of each are financial indicators from the companies. The function `describe()` returns information about the distribution of the data such as `quantiles`, `min` and `max`.


```python
# Statistical summary 
df_rating.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>currentRatio</th>
      <th>quickRatio</th>
      <th>cashRatio</th>
      <th>daysOfSalesOutstanding</th>
      <th>netProfitMargin</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2021.000000</td>
      <td>2021.000000</td>
      <td>2021.000000</td>
      <td>2021.000000</td>
      <td>2021.000000</td>
      <td>...</td>
      <td>2021.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>3.535411</td>
      <td>2.657150</td>
      <td>0.669048</td>
      <td>334.855415</td>
      <td>0.278725</td>
      <td>...</td>
    </tr>
    <tr>
      <td>std</td>
      <td>44.139386</td>
      <td>33.009920</td>
      <td>3.590902</td>
      <td>4456.606352</td>
      <td>6.076128</td>
      <td>...</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-0.932005</td>
      <td>-1.893266</td>
      <td>-0.192736</td>
      <td>-811.845623</td>
      <td>-101.845815</td>
      <td>...</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.071930</td>
      <td>0.602298</td>
      <td>0.131433</td>
      <td>22.806507</td>
      <td>0.020894</td>
      <td>...</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>1.492804</td>
      <td>0.979094</td>
      <td>0.297859</td>
      <td>42.281804</td>
      <td>0.064323</td>
      <td>...</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>2.160710</td>
      <td>1.450457</td>
      <td>0.625355</td>
      <td>59.165369</td>
      <td>0.113871</td>
      <td>...</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1725.505005</td>
      <td>1139.541703</td>
      <td>125.917417</td>
      <td>115961.637400</td>
      <td>198.517873</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 25 columns</p>
</div>



### Skewness and Outliers

We observe a lot of skewness in the data with this first exploration. In this case, it means that most variables in the dataset may strong presence of outliers. Taking as observation the table above the first column:

- `currentRatio`: This 50% of its variables between `1.071` and `2.166891`. The minimum value is `-0.932005` however the maximum value is `1725.505005`. It means, in other words, there is a giant outlier that is extremely distant from most points from the data (currentRatio). 

The same pattern can be observed in the following columns such as `quickRatio`,	`cashRatio`, `daysOfSalesOutstanding`, `netProfitMargin` and so on.

To observe how this reflect on the distribution of the data lets make some plots of variables chose randomly.


```python
column_list = list(df_rating.columns[6:31])
column_list = sample(column_list,4) 
print(column_list)
```

<samp class="text-primary">

    ['operatingCashFlowSalesRatio', 'grossProfitMargin', 'returnOnEquity', 'companyEquityMultiplier']
</samp>


```python
figure, axes = plt.subplots(nrows=2, ncols=4, figsize=(9,5))

axes[0, 0].hist(df_rating[column_list[0]])
axes[0, 1].hist(df_rating[column_list[1]])
axes[1, 0].hist(df_rating[column_list[2]])
axes[1, 1].hist(df_rating[column_list[3]])

axes[0, 2].boxplot(df_rating[column_list[0]])
axes[1, 2].boxplot(df_rating[column_list[1]])
axes[0, 3].boxplot(df_rating[column_list[2]])
axes[1, 3].boxplot(df_rating[column_list[3]])

figure.tight_layout()
```


![png](/assets/images/corporateCreditRatingPrediction-Copy1_24_0.png)


As predicted, the data is comtaminated by outliers. We canot observe real behaviour of the distribution because some points differ too much from the others. We will use the function `.skew` from pandas in all columns. It should return between 0 and 1 if a column is normally distributed. 


```python
df_rating.skew(axis=0)
```

<samp class="text-primary">
    currentRatio                          34.271115<br>
    quickRatio                            30.864610<br>
    cashRatio                             27.046952<br>
    daysOfSalesOutstanding                20.359098<br>
    netProfitMargin                       17.585073<br>
    pretaxProfitMargin                    22.052558<br>
    grossProfitMargin                    -14.198688<br>
    operatingProfitMargin                 26.441502<br>
    returnOnAssets                       -32.049111<br>
    returnOnCapitalEmployed              -33.252701<br>
    returnOnEquity                        31.639845<br>
    assetTurnover                         25.968848<br>
    fixedAssetTurnover                    26.068762<br>
    debtEquityRatio                        0.268074<br>
    debtRatio                              1.284256<br>
    effectiveTaxRate                      32.265705<br>
    freeCashFlowOperatingCashFlowRatio   -22.868222<br>
    freeCashFlowPerShare                  33.610677<br>
    cashPerShare                          33.958646<br>
    companyEquityMultiplier                0.268175<br>
    ebitPerRevenue                        22.055668<br>
    enterpriseValueMultiple               13.920117<br>
    operatingCashFlowPerShare             30.292914<br>
    operatingCashFlowSalesRatio           25.400129<br>
    payablesTurnover                      25.868293<br>
    dtype: float64<br>
</samp>

<br>
<br>
We observe this is a generalized problem. As we can see almost all columns are extremely skewed. We will now go deeper in the investigation of outliers. The following code will return the proportion of outliers in each column . The definition of outlier will be the one from the boxplot - above or bellow `1.5 x IQR`.


```python
for c in df_rating.columns[6:31]:

    q1 = df_rating[c].quantile(0.25)
    q3 = df_rating[c].quantile(0.75)
    iqr = q3 - q1 #Interquartile range
    fence_low  = q3-1.5*iqr
    fence_high = q1+1.5*iqr
    lower_out = len(df_rating.loc[(df_rating[c] < fence_low)  ,c])
    upper_out = len(df_rating.loc[(df_rating[c] > fence_high)  ,c])
    outlier_count = upper_out+lower_out
    prop_out = outlier_count/len(df_rating)
    print(c, ": "+"{:.2%}".format(prop_out))

```

<samp class="text-primary">
    currentRatio : 18.01%<br>
    quickRatio : 19.05%<br>
    cashRatio : 14.84%<br>
    daysOfSalesOutstanding : 23.55%<br>
    netProfitMargin : 25.09%<br>
    pretaxProfitMargin : 24.49%<br>
    grossProfitMargin : 0.99%<br>
    operatingProfitMargin : 22.12%<br>
    returnOnAssets : 24.25%<br>
    returnOnCapitalEmployed : 22.07%<br>
    returnOnEquity : 28.70%<br>
    assetTurnover : 15.83%<br>
    fixedAssetTurnover : 13.46%<br>
    debtEquityRatio : 22.07%<br>
    debtRatio : 21.33%<br>
    effectiveTaxRate : 28.06%<br>
    freeCashFlowOperatingCashFlowRatio : 16.92%<br>
    freeCashFlowPerShare : 23.55%<br>
    cashPerShare : 17.12%<br>
    companyEquityMultiplier : 22.02%<br>
    ebitPerRevenue : 24.34%<br>
    enterpriseValueMultiple : 23.70%<br>
    operatingCashFlowPerShare : 17.66%<br>
    operatingCashFlowSalesRatio : 16.87%<br>
    payablesTurnover : 14.45%<br>
    

Most columns have a significant number of outliers. However it is not clear for us if there are a few rows that all outliers or each of the rows may be contributing individually with some outliers. We will now check by row the distribution of outliers. We will create a new dataframe that `df_rating_outlier` that will be used with this purpose. In this dataframe every cell will 1 one if the corresponding cell is an outlier in `df_raint` and 0 if it is not.


```python
df_rating_outlier = df_rating.copy()

for c in df_rating_outlier.columns[6:31]:
    
    q1 = df_rating_outlier[c].quantile(0.25)
    q3 = df_rating_outlier[c].quantile(0.75)
    iqr = q3 - q1 #Interquartile range
    fence_low  = q3-1.5*iqr
    fence_high = q1+1.5*iqr
    
    for i in range(len(df_rating_outlier)):
        
        if df_rating.loc[i,c] < fence_low or df_rating.loc[i,c] > fence_high: # if Outlier
            
            df_rating_outlier.loc[i,c] = 1
        
        else: # Not Outlier
            df_rating_outlier.loc[i,c] = 0
```



Now we will be able to count how many outliers each row has and plot it. 


```python
df_rating_outlier["total"] = df_rating_outlier.sum(axis=1)
df_rating_outlier.total.hist(bins = 20)
```


![png](/assets/images/corporateCreditRatingPrediction-Copy1_33_1.png)


This is a very interesting plot. We can see that only up to 400 rows don't have any outliers. Most rows have outliers and maybe they will be useful in the further classification tasks. Therefore we see no value in excluding the outliers from the dataset. However we will perform a transformation on the data so we can reduce its negative impact.

#### Data reshaping

We will now perform the following steps in each of the numerical data. 
1. Normalize the data between 0 and 1 (and multiply by 1.000).
2. Apply log on base 10 on each of the variables. 


```python
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

for c in df_rating.columns[6:31]:

    df_rating[[c]] = min_max_scaler.fit_transform(df_rating[[c]].to_numpy())*1000
    df_rating[[c]] = df_rating[c].apply(lambda x: np.log10(x+0.01))
```

### Again the plots


```python
figure, axes = plt.subplots(nrows=2, ncols=4, figsize=(9,5))

axes[0, 0].hist(df_rating[column_list[0]])
axes[0, 1].hist(df_rating[column_list[1]])
axes[1, 0].hist(df_rating[column_list[2]])
axes[1, 1].hist(df_rating[column_list[3]])

axes[0, 2].boxplot(df_rating[column_list[0]])
axes[1, 2].boxplot(df_rating[column_list[1]])
axes[0, 3].boxplot(df_rating[column_list[2]])
axes[1, 3].boxplot(df_rating[column_list[3]])

figure.tight_layout()
```


![png](/assets/images/corporateCreditRatingPrediction-Copy1_37_0.png)


We have a problem with respect to vizualisation of the data. The impact of the outliers is so big that we cannot observe the patterns in the data. To enhance our visualization we will from now ignore outliers. We will replace then by values with lower impact such as the lower hinge. In this way we will be able to continue with our EDA. To preserve our dataset we will use a new table called `df_rating_no_out`.


```python
df_rating_no_out = df_rating.copy()

for c in df_rating_no_out.columns[6:31]:

    q05 = df_rating_no_out[c].quantile(0.10)
    q95 = df_rating_no_out[c].quantile(0.90)
    iqr = q95 - q05 #Interquartile range
    fence_low  = q05-1.5*iqr
    fence_high = q95+1.5*iqr
    df_rating_no_out.loc[df_rating_no_out[c] > fence_high,c] = df_rating_no_out[c].quantile(0.25)
    df_rating_no_out.loc[df_rating_no_out[c] < fence_low,c] = df_rating_no_out[c].quantile(0.75)
    
```

Now that we have this dataframe we can use it use it to observe the data from a different angle. We will be able to observe the distribution that was hidden by the outliers. The first step:
   - Plot all columns (boxplot) by each label:`High Risk`, `Low Risk`, `Medium Risk`, `Highest Risk`.


```python
figure, axes = plt.subplots(nrows=8, ncols=3, figsize=(20,44))

i = 0 
j = 0

for c in df_rating_no_out.columns[6:30]:
    
    sns.boxplot(x=df_rating_no_out.Rating, y=df_rating_no_out[c], palette="Set3", ax=axes[i, j])
    
    if j == 2:
        j=0
        i+=1
    else:
        j+=1    

```


![png](/assets/images/corporateCreditRatingPrediction-Copy1_41_0.png)


The most interesting point about the previous plots is the fact that they clearly show a difference in the medians and distribution according to the rating (Risk). It points to a scenario where the variables will have good predictive power for classification. Following with our analysis we will create scatter plots to see if we can observe who the variables relate to each other and how labels can be observer in respect to it.


```python
df_rating.colors = 'a'
df_rating_no_out.loc[df_rating_no_out['Rating'] == 'Lowest Risk', 'color'] = 'r'
df_rating_no_out.loc[df_rating_no_out['Rating'] == 'Low Risk', 'color'] = 'g'
df_rating_no_out.loc[df_rating_no_out['Rating'] == 'Medium Risk', 'color'] = 'b'
df_rating_no_out.loc[df_rating_no_out['Rating'] == 'High Risk','color'] = 'y'
df_rating_no_out.loc[df_rating_no_out['Rating'] == 'Highest Risk', 'color'] = 'm'
```


```python
column_list = list(df_rating.columns[6:31])
column_list = sample(column_list,12) 
```


```python
figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(14,14))

i = 0 
j = 0

for c in range(0,12, 2):

    sns.scatterplot(x = column_list[c], y=column_list[c+1], hue="color", data=df_rating_no_out, ax=axes[j,i])
    
    if i == 1:
        i = 0
        j +=1
    
    else:
        i+=1
```


![png](/assets/images/corporateCreditRatingPrediction-Copy1_45_0.png)


In fact, we are working with a dataset that has a big numer of dimensions. With two variables it would not be possible to make any predictions. However this is not the case. Unfortunately we are not able to vizualise the data in all its dimensions, but luckely we will be able to perform accurate classificaitons. 

# Machine Learning 

Is it possible to predict what creidt profile a company will receive from a rating agency based on its financial indicators? If so, what are the most important predictors? Apparently not much work has been done with regards to this question. This academic [paper](https://www.researchgate.net/publication/331386740_Credit_Rating_Forecasting_Using_Machine_Learning_Techniques) was the only work found about it. It is worth checking it out. As we will do it, it tests most ML algorithms and identifies the most important features. 

In the following steps we will perform the following:

1. Prepare the dataset 
    - Split in train and test
    - Transform/Encode the features kand labels
2. Test a wide range of ML models (Tree-based, Probabilistic and so on). 
3. Compare the accuracry of all models. 
4. Choose our winning model and tune hyperparameters to target a higher accuracy.
5. Make a more profound evaluation of the result with a confusion matrix and different measures. 
6. identify the most important features to predict the rating. 



## Prepare the Dataset


```python
le = preprocessing.LabelEncoder()
le.fit(df_rating.Sector)
df_rating.Sector = le.transform(df_rating.Sector) # encode sector
le.fit(df_rating.Rating)
df_rating.Rating = le.transform(df_rating.Rating) # encode rating
```


```python
df_train, df_test = train_test_split(df_rating, test_size=0.2, random_state = 1234)
```


```python
X_train, y_train = df_train.iloc[:,5:31], df_train.iloc[:,0]
X_test, y_test = df_test.iloc[:,5:31], df_test.iloc[:,0]
```

# Fit Models

Now we will test a range of models. In each we will fit the model in the train data, make predictons for the test data and  obtain the accuracy. In later steps we will compare the accuracy of all the models. We will use primarily the library `sklearn` but also `XGBoost`.  

#### XGBoost


```python
XGB_model = xgb.XGBRegressor(objective ='multi:softmax', num_class =4)
XGB_model.fit(X_train, y_train)
y_pred_XGB = XGB_model.predict(X_test)
Accuracy_XGB = metrics.accuracy_score(y_test, y_pred_XGB)
print("XGB Accuracy:",Accuracy_XGB)
```

<div class="p-3 mb-2 bg-dark text-white">XGB Accuracy: 0.691358024691358</div>
<br><br>    

#### Gradient Boosting Classifier


```python
GBT_model = GradientBoostingClassifier(random_state=123)
GBT_model.fit(X_train, y_train)
y_pred_GBT = GBT_model.predict(X_test)
Accuracy_GBT = metrics.accuracy_score(y_test, y_pred_GBT)
print("GBT Accuracy:",Accuracy_GBT)
```

<div class="p-3 mb-2 bg-dark text-white">GBT Accuracy: 0.6320987654320988</div>
<br><br>

#### Random Forest


```python
RF_model = RandomForestClassifier(random_state=1234)
RF_model.fit(X_train,y_train)
y_pred_RF = RF_model.predict(X_test)
Accuracy_RF = metrics.accuracy_score(y_test, y_pred_RF)
print("RF Accuracy:",Accuracy_RF)
```

<div class="p-3 mb-2 bg-dark text-white">RF Accuracy: 0.6246913580246913</div>
<br><br>


#### Support Vector Machine



```python
SVC_model = svm.SVC(kernel='rbf', gamma= 2, C = 5, random_state=1234)
SVC_model.fit(X_train, y_train)
y_pred_SVM = SVC_model.predict(X_test)
Accuracy_SVM = metrics.accuracy_score(y_test, y_pred_SVM)
print("SVM Accuracy:",Accuracy_SVM)
```

<div class="p-3 mb-2 bg-dark text-white">SVM Accuracy: 0.5333333333333333</div>
<br><br>


#### Neural Network


```python
MLP_model = MLPClassifier(hidden_layer_sizes=(5,5,5), activation='logistic', solver='adam', max_iter=1500)
MLP_model.fit(X_train, y_train)
y_pred_MLP = MLP_model.predict(X_test)
Accuracy_MLP = metrics.accuracy_score(y_test, y_pred_MLP)
print("MLP Accuracy:",Accuracy_MLP)
```

<div class="p-3 mb-2 bg-dark text-white">MLP Accuracy: 0.3654320987654321</div>
<br><br>


#### Naive Bayes


```python
GNB_model = GaussianNB()
GNB_model.fit(X_train, y_train)
y_pred_GNB = GNB_model.predict(X_test)
Accuracy_GNB = metrics.accuracy_score(y_test, y_pred_GNB)
print("GNB Accuracy:",Accuracy_GNB)
```

<div class="p-3 mb-2 bg-dark text-white">GNB Accuracy: 0.30864197530864196</div>
<br><br>


#### Linear Discriminant Analysis


```python
LDA_model = LinearDiscriminantAnalysis()
LDA_model.fit(X_train,y_train)
y_pred_LDA = LDA_model.predict(X_test)
Accuracy_LDA = metrics.accuracy_score(y_test, y_pred_LDA)
print("LDA Accuracy:",Accuracy_LDA)
```

<div class="p-3 mb-2 bg-dark text-white">LDA Accuracy: 0.38765432098765434</div>
<br><br>


#### Quadratic Discriminant Analysis


```python
QDA_model = QuadraticDiscriminantAnalysis()
QDA_model.fit(X_train,y_train)
y_pred_QDA = QDA_model.predict(X_test)
Accuracy_QDA = metrics.accuracy_score(y_test, y_pred_QDA)
print("QDA Accuracy:",Accuracy_QDA)
```    

<div class="p-3 mb-2 bg-dark text-white">QDA Accuracy: 0.35555555555555557</div>
<br><br>


#### K Nearest Neighbours



```python
KNN_model = KNeighborsClassifier(n_neighbors = 3)
KNN_model.fit(X_train,y_train)
y_pred_KNN = KNN_model.predict(X_test)
Accuracy_KNN = metrics.accuracy_score(y_test, y_pred_KNN)
print("KNN Accuracy:",Accuracy_KNN)
```

<div class="p-3 mb-2 bg-dark text-white">KNN Accuracy: 0.5802469135802469</div>
<br><br>


#### Logistic Regression



```python
LR_model = LogisticRegression(random_state=1234 , multi_class='multinomial', solver='newton-cg')
LR_model = LR_model.fit(X_train, y_train)
y_pred_LR = LR_model.predict(X_test)
Accuracy_LR = metrics.accuracy_score(y_test, y_pred_LR)
print("LR Accuracy:",Accuracy_LR)
```

<div class="p-3 mb-2 bg-dark text-white">LR Accuracy: 0.3925925925925926</div>
<br><br>


## Compare Results


```python
accuracy_list = [Accuracy_XGB, Accuracy_GBT, Accuracy_RF, Accuracy_SVM, Accuracy_MLP, Accuracy_GNB, 
                 Accuracy_LDA, Accuracy_QDA, Accuracy_KNN, Accuracy_LR]

model_list = ['XGBboost', 'Gradient Boosting', 'Random Forest', 'Support Vector Machine', 
              "Neural Network", 'Naive Bayes', 'Linear Discriminat', 'Quadratic Discriminat', 
              'KNN', 'Logistic Regression']

df_accuracy = pd.DataFrame({'Model': model_list, 'Accuracy': accuracy_list})
```


```python
order = list(df_accuracy.sort_values('Accuracy', ascending=False).Model)
df_accuracy = df_accuracy.sort_values('Accuracy', ascending=False).reset_index().drop(['index'], axis=1)

plt.figure(figsize=(12,8))
# make barplot and sort bars
x = sns.barplot(x='Model', y="Accuracy", data=df_accuracy, order = order, palette="rocket")
plt.xlabel("Model", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.title("Accuracy by Model", fontsize=20)
plt.grid(linestyle='-', linewidth='0.5', color='grey')
plt.xticks(rotation=70, fontsize=12)
plt.ylim(0,1)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

for i in range(len(model_list)):
    plt.text(x = i, y = df_accuracy.loc[i, 'Accuracy'] + 0.05, s = str(round((df_accuracy.loc[i, 'Accuracy'])*100, 2))+'%', 
             fontsize = 14, color='black',horizontalalignment='center')

y_value=['{:,.2f}'.format(x) + '%' for x in ax.get_yticks()]
ax.set_yticklabels(y_value)

plt.tight_layout()

```


![png](/assets/images/corporateCreditRatingPrediction-Copy1_73_0.png)


We have our winner. XGboost is the best performing model. 

## XGBoost Hyperparameter Tunning

The XGboost model has achieved a very high accuracy given that we have 4 different classes. Now we will try to increase the performance even more. We will use a cross-validation approach and we will follow similar steps to this [tutorial](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f). First we load the train and test data into DMatrices. `DMatrix` is a data structure used by XGBoost to optimize both memory efficiency and training speed.


```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
```

#### The params dictionary
We create a dictonary with the parameters from our previous XGboost model.


```python
params = XGB_model.get_xgb_params()
```

```json
    {'objective': 'multi:softmax',
     'base_score': 0.5,
     'booster': 'gbtree',
     'colsample_bylevel': 1,
     'colsample_bynode': 1,
     'colsample_bytree': 1,
     'gamma': 0,
     'gpu_id': -1,
     'interaction_constraints': '',
     'learning_rate': 0.300000012,
     'max_delta_step': 0,
     'max_depth': 6,
     'min_child_weight': 1,
     'monotone_constraints': '()',
     'n_jobs': 0,
     'num_parallel_tree': 1,
     'random_state': 0,
     'reg_alpha': 0,
     'reg_lambda': 1,
     'scale_pos_weight': None,
     'subsample': 1,
     'tree_method': 'exact',
     'validate_parameters': 1,
     'verbosity': None,
     'num_class': 4}
```


We will use the `merror` error parameter from classification. It is basic an accuracy. 


```python
params['eval_metric'] = "merror"
```

The num_boost_round which corresponds to the maximum number of boosting rounds that we allow. 


```python
num_boost_round = 1000
```


```python
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50,
    verbose_eval=30)

print("Best merror: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))
```



    [0]	Test-merror:0.44691<br>
    Will train until Test-merror hasn't improved in 50 rounds.<br>
    [30]	Test-merror:0.34568<br>
    [60]	Test-merror:0.32839<br>
    [90]	Test-merror:0.30617<br>
    [120]	Test-merror:0.30864<br>
    [150]	Test-merror:0.30864<br>
    Stopping. Best iteration:<br>
    [104]	Test-merror:0.29877<br>
    
    Best merror: 0.30 with 105 rounds
    

#### Using XGBoost’s CV

In order to tune the other hyperparameters, we will use the cv function from XGBoost. It allows us to run cross-validation on our training dataset and returns a mean merror score. We will use a `k = 5` for every parameter.



```python
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'merror'},
    early_stopping_rounds=50,
    verbose_eval=30
)
cv_results.tail()
```

    [0]	train-merror:0.24412+0.00946	test-merror:0.44677+0.01983
    [30]	train-merror:0.00015+0.00031	test-merror:0.35580+0.04214
    [60]	train-merror:0.00000+0.00000	test-merror:0.34466+0.03896
    [90]	train-merror:0.00000+0.00000	test-merror:0.34775+0.04295
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train-merror-mean</th>
      <th>train-merror-std</th>
      <th>test-merror-mean</th>
      <th>test-merror-std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>55</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.347756</td>
      <td>0.036916</td>
    </tr>
    <tr>
      <td>56</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.345899</td>
      <td>0.036946</td>
    </tr>
    <tr>
      <td>57</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.345284</td>
      <td>0.039353</td>
    </tr>
    <tr>
      <td>58</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.343426</td>
      <td>0.038823</td>
    </tr>
    <tr>
      <td>59</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.342811</td>
      <td>0.037624</td>
    </tr>
  </tbody>
</table>
</div>




```python
cv_results['test-merror-mean'].min()
```




    0.34281059999999997



Now we are ready to start tuning. We will first tune our parameters to minimize the merror on cross-validation, and then check the performance of our model on the test dataset. 

#### Parameters `max_depth` and `min_child_weight`. 


```python
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(5,12)
    for min_child_weight in range(5,8)
]
```


```python
# Define initial best params and MAE
min_merror = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'merror'},
        early_stopping_rounds=50,
        verbose_eval=False

    )
    # Update best merror
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, merror: {}".format(best_params[0], best_params[1], min_merror))
```

    CV with max_depth=5, min_child_weight=5
    	MAE 0.3533254 for 75 rounds
    CV with max_depth=5, min_child_weight=6
    	MAE 0.3533292 for 132 rounds
    CV with max_depth=5, min_child_weight=7
    	MAE 0.3588922 for 100 rounds
    CV with max_depth=6, min_child_weight=5
    	MAE 0.3496082 for 84 rounds
    CV with max_depth=6, min_child_weight=6
    	MAE 0.3508504 for 165 rounds
    CV with max_depth=6, min_child_weight=7
    	MAE 0.3545598 for 79 rounds
    CV with max_depth=7, min_child_weight=5
    	MAE 0.33784559999999997 for 82 rounds
    CV with max_depth=7, min_child_weight=6
    	MAE 0.3471276 for 220 rounds
    CV with max_depth=7, min_child_weight=7
    	MAE 0.3533136 for 45 rounds
    CV with max_depth=8, min_child_weight=5
    	MAE 0.3446568 for 70 rounds
    CV with max_depth=8, min_child_weight=6
    	MAE 0.3514716 for 45 rounds
    CV with max_depth=8, min_child_weight=7
    	MAE 0.35703860000000004 for 95 rounds
    CV with max_depth=9, min_child_weight=5
    	MAE 0.3409416 for 80 rounds
    CV with max_depth=9, min_child_weight=6
    	MAE 0.3384664 for 60 rounds
    CV with max_depth=9, min_child_weight=7
    	MAE 0.34961180000000003 for 60 rounds
    CV with max_depth=10, min_child_weight=5
    	MAE 0.3458856 for 46 rounds
    CV with max_depth=10, min_child_weight=6
    	MAE 0.3403222 for 135 rounds
    CV with max_depth=10, min_child_weight=7
    	MAE 0.34712400000000004 for 43 rounds
    CV with max_depth=11, min_child_weight=5
    	MAE 0.3390972 for 117 rounds
    CV with max_depth=11, min_child_weight=6
    	MAE 0.34837559999999995 for 81 rounds
    CV with max_depth=11, min_child_weight=7
    	MAE 0.34157399999999993 for 134 rounds
    Best params: 7, 5, merror: 0.33784559999999997
    

We get the best score with a max_depth of 9 and min_child_weight of 6, so let's


```python
params['max_depth'] = 7
params['min_child_weight'] = 5
```

#### Parameters `subsample` and `colsample_bytree`


```python
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]
```


```python
# Define initial best params and MAE
min_merror = float("Inf")
best_params = None

for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'merror'},
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Update best MAE
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = (subsample,colsample)
print("Best params: {}, {}, merror: {}".format(best_params[0], best_params[1], min_mae))
```

    CV with subsample=1.0, colsample=1.0
    	MAE 0.34280299999999997 for 57 rounds
    CV with subsample=1.0, colsample=0.9
    	MAE 0.35086200000000006 for 47 rounds
    CV with subsample=1.0, colsample=0.8
    	MAE 0.335384 for 53 rounds
    CV with subsample=1.0, colsample=0.7
    	MAE 0.3477508 for 46 rounds
    CV with subsample=0.9, colsample=1.0
    	MAE 0.3489852 for 47 rounds
    CV with subsample=0.9, colsample=0.9
    	MAE 0.36075179999999996 for 32 rounds
    CV with subsample=0.9, colsample=0.8
    	MAE 0.35519259999999997 for 22 rounds
    CV with subsample=0.9, colsample=0.7
    	MAE 0.3316686 for 67 rounds
    CV with subsample=0.8, colsample=1.0
    	MAE 0.344668 for 22 rounds
    CV with subsample=0.8, colsample=0.9
    	MAE 0.344045 for 31 rounds
    CV with subsample=0.8, colsample=0.8
    	MAE 0.3490044 for 49 rounds
    CV with subsample=0.8, colsample=0.7
    	MAE 0.356431 for 41 rounds
    CV with subsample=0.7, colsample=1.0
    	MAE 0.34713720000000003 for 51 rounds
    CV with subsample=0.7, colsample=0.9
    	MAE 0.352079 for 66 rounds
    CV with subsample=0.7, colsample=0.8
    	MAE 0.34591779999999994 for 65 rounds
    CV with subsample=0.7, colsample=0.7
    	MAE 0.3521078 for 57 rounds
    Best params: 0.9, 0.7, merror: 0.3316686
    


```python
params['subsample'] =0.9
params['colsample_bytree'] = 0.7
```

#### Parameter `ETA`



```python
%time
# This can take some time…
min_merror = float("Inf")
best_params = None

for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['merror'],
            early_stopping_rounds=10
)
    # Update best score
    mean_mae = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, merror: {}".format(best_params, min_mae))
```

    Wall time: 0 ns
    CV with eta=0.3
    	MAE 0.3316686 for 67 rounds
    
    CV with eta=0.2
    	MAE 0.3316686 for 67 rounds
    
    CV with eta=0.1
    	MAE 0.3316686 for 67 rounds
    
    CV with eta=0.05
    	MAE 0.3316686 for 67 rounds
    
    CV with eta=0.01
    	MAE 0.3316686 for 67 rounds
    
    CV with eta=0.005
    	MAE 0.3316686 for 67 rounds
    
    Best params: None, merror: 0.3316686
    


```python
params['eta'] = .3
```

#### Results

This are the final parameters of our tunned model.


```python
params
```




    {'objective': 'multi:softmax',
     'base_score': 0.5,
     'booster': 'gbtree',
     'colsample_bylevel': 1,
     'colsample_bynode': 1,
     'colsample_bytree': 0.7,
     'gamma': 0,
     'gpu_id': -1,
     'interaction_constraints': '',
     'learning_rate': 0.300000012,
     'max_delta_step': 0,
     'max_depth': 7,
     'min_child_weight': 5,
     'monotone_constraints': '()',
     'n_jobs': 0,
     'num_parallel_tree': 1,
     'random_state': 0,
     'reg_alpha': 0,
     'reg_lambda': 1,
     'scale_pos_weight': None,
     'subsample': 0.9,
     'tree_method': 'exact',
     'validate_parameters': 1,
     'verbosity': None,
     'num_class': 4,
     'eval_metric': 'merror',
     'eta': 0.3}



Let’s train a model with it and see how well it does on our test set!


```python
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=1000,
    verbose_eval=100
)
```

    [0]	Test-merror:0.45679
    Will train until Test-merror hasn't improved in 1000 rounds.
    [100]	Test-merror:0.34321
    [200]	Test-merror:0.33827
    [300]	Test-merror:0.34321
    [400]	Test-merror:0.34321
    [500]	Test-merror:0.34815
    [600]	Test-merror:0.35062
    [700]	Test-merror:0.34815
    [800]	Test-merror:0.34815
    [900]	Test-merror:0.35062
    [999]	Test-merror:0.35556
    


```python
num_boost_round = model.best_iteration + 1
best_model = xgb.train(
    params,
    dtrain,
    verbose_eval=100,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")]
)
```

    [0]	Test-merror:0.45679
    [100]	Test-merror:0.34321
    [200]	Test-merror:0.33827
    [278]	Test-merror:0.32593
    


```python
metrics.accuracy_score(best_model.predict(dtest), y_test)
```




    0.674074074074074



We did not arrive in an enhanced model with this tunning. Anyone is welcome to continue this tunning and achieve a superior accuracy.

## Confusion Matrix

We will now analyse according to each class the performance of the model. The best way to do it is with a confusion matrix. We can see how many points were missclassified and where were then classified to if not the right rating.


```python
cm = confusion_matrix(y_test, y_pred_XGB)
```


```python
fig, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm, annot = True, ax = ax, vmin=0, vmax=150, fmt="d", linewidths=.5, linecolor = 'white', cmap="Reds") # annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Medium Risk','Highest Risk', 'Low Risk', 'High Risk'])
ax.yaxis.set_ticklabels(['Medium Risk','Highest Risk', 'Low Risk', 'High Risk']);

# This part is to correct a bug from the heatmap funciton from pyplot
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
```


![png](/assets/images/corporateCreditRatingPrediction-Copy1_108_0.png)


#### Analysis
Given the fact that the dataset is very unbalanced, with have achieved a very low accuracy (actually 0) for very risky companies. To deal with it we wiould have to apply upsampling techniques which we may in a future work. Now we analyse other metrics as Precision, recall and F1 from our targets.


```python
print(classification_report(y_test, y_pred_XGB, target_names = ['Medium Risk','Highest Risk', 'Low Risk', 'High Risk']))
```

                  precision    recall  f1-score   support
    
     Medium Risk       0.72      0.83      0.77       148
    Highest Risk       0.00      0.00      0.00         8
        Low Risk       0.73      0.67      0.70       107
       High Risk       0.64      0.60      0.62       142
    
        accuracy                           0.69       405
       macro avg       0.52      0.53      0.52       405
    weighted avg       0.68      0.69      0.68       405
    
    

Apparetly the fact that we have more labels in the edium Risk has enhanced its classification. However, overall we have achieved good classification scaores for most, with the exception of Highest Risk.


## Feature Selection
In our tast task we will identify which features were the most valuable for our model. In our first step we will check if by any chance we can increase the accuracy of our model extracting a feature.


```python
thresholds = sort(XGB_model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(XGB_model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
```

    Thresh=0.025, n=26, Accuracy: 69.14%
    Thresh=0.026, n=25, Accuracy: 67.16%
    Thresh=0.026, n=24, Accuracy: 67.16%
    Thresh=0.027, n=23, Accuracy: 64.69%
    Thresh=0.027, n=22, Accuracy: 64.44%
    Thresh=0.027, n=21, Accuracy: 65.93%
    Thresh=0.028, n=20, Accuracy: 64.44%
	...
	
<br>	
It is not the case. Now lets visualize which are the most relevant features. 


```python
from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(8, 8))
# xgboost.plot_importance(..., ax=ax)

plot_importance(model, ax=ax)
plt.show()
```


![png](/assets/images/corporateCreditRatingPrediction-Copy1_114_0.png)


# Visualize Companies

**Bonus**: In this dataset we are working exclusevely with companies that are traded in the stock exchanges from the US. 
Now, we will visualize which companies are considered secure to lend money according to agencies. We will make 4 different wordclouds, one for each rating of risk. 
 
 #### Create a function to generate text for the word cloud
 


```python
def WCloud(dataframe, column, rating):
    
    words = ''
    
    # iterate through the csv file 
    for val in dataframe.loc[dataframe['Rating'] == rating, column]:
      
        # typecaste each val to string 
        val = str(val)
        val = val.replace(".", "")
        val = val.replace(",", "")

        # split the value 
        tokens = val.split()

        #Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        words += " ".join(tokens) + " "
        
    return words
```


![png](/assets/images/corporateCreditRatingPrediction-Copy1_124_1.png)


You may observe that some companies may be in different plots. Thats because they have been rated in different times with different rates.
