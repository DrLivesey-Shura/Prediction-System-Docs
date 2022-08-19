# Example with JSON

## Restaurent recommendation system with Machine Learning using the Python programming language.

* Importing some libraries:

```python
from sklearn.preprocessing import MinMaxScaler
import re
import numpy as np
from nltk.corpus import stopwords
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from recommendation_system import SysRecommenadtion, SysRecMethode
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
import sqlalchemy as db
import pymongo
```

* importing the datasets:

```python
with open('zomato_light.json', encoding='utf-8-sig') as f_input:
    df = pd.read_json(f_input)
```

* Normalizing the dataset:  

```python
# Deleting Unnnecessary Columns
# Dropping the column "dish_liked", "phone", "url" and saving the new dataset as "df2"
df2 = df.drop(['url', 'dish_liked', 'phone'], axis=1)

# Removing the Duplicates
df2.duplicated().sum()
df2.drop_duplicates(inplace=True)

# Remove the NaN values from the dataset
df2.isnull().sum()
df2.dropna(how='any', inplace=True)

# Changing the column names
df2 = df2.rename(columns={'approx_cost(for two people)': 'cost',
                          'listed_in(type)': 'type', 'listed_in(city)': 'city'})

# # Some Transformations
# df2['cost'] = df2['cost'].astype(str)  # Changing the cost to string
# # Using lambda function to replace ',' from cost
# df2['cost'] = df2['cost'].apply(lambda x: x.replace(',', '.'))
# df2['cost'] = df2['cost'].astype(float)

# Some Transformations
df2['cost'] = df2['cost'].astype(str)  # Changing the cost to string
df2['cost'] = df2['cost'].replace(',','.', regex=True).astype(float)

# Removing '/5' from Rates
df2 = df2.loc[df2.rate != 'NEW']
df2 = df2.loc[df2.rate != '-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == str else x

df2.rate = df2.rate.apply(remove_slash).str.strip().astype('float')

# Adjust the column names
df2.name = df2.name.apply(lambda x: x.title())
df2.online_order.replace(('Yes', 'No'), (True, False), inplace=True)
df2.book_table.replace(('Yes', 'No'), (True, False), inplace=True)

# Computing Mean Rating
restaurants = list(df2['name'].unique())
df2['Mean Rating'] = 0

for i in range(len(restaurants)):
    df2.loc[df2['name'] == restaurants[i],
            'Mean Rating'] = df2['rate'][df2['name'] == restaurants[i]].mean()
scaler = MinMaxScaler(feature_range=(1,5))
df2[['Mean Rating']] = scaler.fit_transform(
    df2[['Mean Rating']]).round(2)

# Lower Casing

df2["reviews_list"] = df2["reviews_list"].str.lower()

# Removal of Puctuations
PUNCT_TO_REMOVE = string.punctuation


def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


df2["reviews_list"] = df2["reviews_list"].apply(
    lambda text: remove_punctuation(text))

# Removal of Stopwords
STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


df2["reviews_list"] = df2["reviews_list"].apply(
    lambda text: remove_stopwords(text))

# Removal of URLS


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


df2["reviews_list"] = df2["reviews_list"].apply(
    lambda text: remove_urls(text))

df2[['reviews_list', 'cuisines']].sample(5)

# RESTAURANT NAMES:
restaurant_names = list(df2['name'].unique())


def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range=nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]


df2 = df2.drop(['address', 'rest_type', 'type',
                'menu_item', 'votes'], axis=1)

# Randomly sample 60% of your dataframe
df_percent = df2.sample(frac=0.5)
```

* Preparing the model: 

```python
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])
sr.build(tfidf_matrix, SysRecMethode.TFIDF)
```

* Making the system recomendation: 

```
sr = SysRecommenadtion(df2)
sr.indices = pd.Series(
    df2.index, index=df2["name"]).drop_duplicates()
sr.threshold = 10
```

* Providing the restaurent movies for the given restaurent name: 

```
s = sr.predict(value='Spice Elephant' ,keys=['cuisines','Mean Rating', 'cost'])
print(s)
```

* Output:

```md
                            cuisines                Mean Rating     cost
1                  Chinese, North Indian, Thai         3.89         800.0
2                       Cafe, Mexican, Italian         3.22         800.0
3                   South Indian, North Indian         3.00         300.0
4                     North Indian, Rajasthani         3.22         600.0
5                                 North Indian         3.22         600.0
6  North Indian, South Indian, Andhra, Chinese         2.78         800.0
7                         Pizza, Cafe, Italian         5.00         600.0
8                   Cafe, Italian, Continental         3.67         700.0
9     Cafe, Mexican, Italian, Momos, Beverages         4.11         550.0
```