# Example with CSV

## Movies recommendation system with Machine Learning using the Python programming language.

* Importing some libraries:

```python
from __future__ import print_function
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from recommendation_system import SysRecommenadtion, SysRecMethode
from sklearn.feature_extraction.text import TfidfVectorizer
import pymongo
```

* importing the datasets:

```python
df1 = pd.read_csv('./tmdb_5000_credits.csv')
df2 = pd.read_csv('./tmdb_5000_movies.csv')
```


* Mergine the datasets:

```python
df1.columns = ['id', 'tittle', 'cast', 'crew'] 
df2 = pd.concat([df2 ,df1]) 
```

* Preparing the model: 

```python
tfidf = TfidfVectorizer(stop_words='english')

df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
tfidf_matrix.shape
```

* Making the system recomendation: 

```python
sr = SysRecommenadtion(df2)
sr.build(tfidf_matrix, SysRecMethode.TFIDF)
sr.indices = pd.Series(
    df2.index, index=df2["title"]).drop_duplicates()

sr.threshold = 10
```

* Providing the recomanded movies for the given movie: 

```python
s = sr.predict(key='title', value='Avatar')
print(s)
```

* Output:

```md
1    Pirates of the Caribbean: At World's End
2                                     Spectre
3                       The Dark Knight Rises
4                                 John Carter
5                                Spider-Man 3
6                                     Tangled
7                     Avengers: Age of Ultron
8      Harry Potter and the Half-Blood Prince
9          Batman v Superman: Dawn of Justice
Name: title, dtype: object
```