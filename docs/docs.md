# Explaining the main calss

In Machine Learning, there is an extended class of web applications that involve predicting user responses to options. Such an installation is called a recommender system.

Recommendation systems are widely used today to recommend products to users based on their interests. A recommendation system is one of
the strongest systems for increasing profits by retaining more users in a very big competition. In the previous examples i showed u how 
to build system Recommendation for multiple type of file (CSV , Json , MongoDB , SqlAlchemy) and using different module (TFIDF and KNN)
using only one class(wraper) that can fit to all this type of files and modules

# The Recommendation system class

The class `SysRecommenadtion` take only the Dataset as a parameter `SysRecommenadtion(dataset_name)` and different methode such as:

## The Build methode:

``build(model , SysRecMethode)`` 

- The ``model`` which it is the model for recommendation we have built 

- and the ``SysRecMethode`` its the type of algorithme we chouse (TFIDF or KNN) and if not chousen it will take the KNN algorithme as a default value

So now if the KNN algorithme is chousen the methode ``features`` is gona be required so it can fit with the model using the KNN    algorithme it will make the build and get a list of the cousine simliarities which is gona be use in the prediction of the    Recommendation system

And now if the TFIDF algorithme is been used nothing is recuired but the model which is gona be the TFIDF matrix using the ``TfidfVectorizer`` and the ``tfidf.fit_transform`` to build and get a list of the cousine simliarities which is gona be use in the prediction of the Recommendation system

## The Prediction methode:

When this function is called, we will have to pass the value (Movie title , restaurent name ....) to it. The model will try to find results based on the features. We’ll store those results that the system recommends in a list and return them at the end:

``predict(value, key, keys)``

- The predict methode is gonna return a list of Recommendation based on the ``value`` given (Movie title , restaurent name ....) and the ``key`` is the column name you wan to show  (the title or the name for example) and if u wan to show multiple column at one use the ``keys`` parameter and give it a list of the column to show 

* ``predict(value='Spice Elephant' ,keys=['cuisines','Mean Rating', 'cost'])``

First of all the key or the keys must be provided so it can do the prediction and get the list of Recommendation and then we hae two choices:

- Using the KNN algorithme:
    We use the KNN algorithm to build our Recommendation system with Machine Learning using Python by getting the id from the source which have the same id's as the value given and return the list of results based on the value given

- Using the TFIDF algorithme:
    We use the TFIDF algorithm to build our Recommendation system with Machine Learning using Python using the indice of the value given to get the similiaries result based on the value given by using the cosins similiarities and return the list of recomendation 

- note: 

    The ``indices`` methode must be provided for the TFIDF algorithme so its can do the predection and get the result list

    Use The ``threshold`` methode to provide the number of the results u wan to show 

