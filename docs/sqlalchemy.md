# Example with SqlAlchemy

## Books recommendation system with Machine Learning using the Python programming language.

* Importing some libraries:

```python
import pandas as pd
import sqlalchemy as db
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from recommendation_system import SysRecommenadtion, SysRecMethode
from sqlalchemy import create_engine
import pymongo
```

* Create the engine:

```python
engine = db.create_engine(
    "mysql://username:password@connection_name:port/databse_name", echo=False)
conn = engine.connect()
```

* Select the table from the database:

```python
stds = db.Table('table_name', metadata, autoload=True, autoload_with=engine)

query = db.select([stds])
ResultProxy = conn.execute(query)
ResultSet = ResultProxy.fetchall()

df = pd.DataFrame(ResultSet)

```

* Normalizing the dataset: 

```python
df = df.interpolate()

df2 = df.copy()
df2.loc[(df2['average_rating'] >= 0) & (df2['average_rating'] <= 1),
        'rating_between'] = "between 0 and 1"
df2.loc[(df2['average_rating'] > 1) & (df2['average_rating'] <= 2),
        'rating_between'] = "between 1 and 2"
df2.loc[(df2['average_rating'] > 2) & (df2['average_rating'] <= 3),
        'rating_between'] = "between 2 and 3"
df2.loc[(df2['average_rating'] > 3) & (df2['average_rating'] <= 4),
        'rating_between'] = "between 3 and 4"
df2.loc[(df2['average_rating'] > 4) & (df2['average_rating'] <= 5),
        'rating_between'] = "between 4 and 5"

rating_df = pd.get_dummies(df2['rating_between'])
language_df = pd.get_dummies(df2['language_code'])

features = pd.concat(
    [rating_df, language_df, df2['average_rating'], df2['ratings_count']], axis=1)

```

* Preparing the model: 

```python
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)
```

* Making the system recomendation: 

```python
sr = SysRecommenadtion(df2)
sr.delNan = True
sr.features = features
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')

sr.build(model, SysRecMethode.KNN)

```

* Providing the recomanded books for the given book title: 

```python
s = sr.predict(
    key='title', value="The Ultimate Hitchhiker's Guide (Hitchhiker's Guide to the Galaxy  #1-5)")
print(s)
```

* Output:

```md
[
    "The Ultimate Hitchhiker's Guide (Hitchhiker's Guide to the Galaxy  #1-5)", 
    "The Complete Monty Python's Flying Circus: All the Words: Volume 1", 
    'Black Rednecks and White Liberals', 
    'Blue at the Mizzen (Aubrey & Maturin #20)', 
    'Azumanga Daioh  Vol. 3 (Azumanga Daioh  #3)',
    "The Ultimate Hitchhiker's Guide: Five Complete Novels and One Story (Hitchhiker's Guide to the Galaxy  #1-5)"
]
```