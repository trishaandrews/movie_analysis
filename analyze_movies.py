import pickle
import pandas as pd
import numpy as np
from patsy import dmatrices

PICKLEDIR = "./pkls/"

def pickle_stuff(filename, data):
    with open(filename, 'w') as picklefile:
        pickle.dump(data, picklefile)

def unpickle(filename):
    with open(filename, 'r') as picklefile:
        old_data = pickle.load(picklefile)
    return old_data

movies = unpickle(PICKLEDIR + "moviesgenre.pkl")

df = pd.DataFrame.from_items(movies.items(), orient='index', 
                               columns=["OriginC", "Budget", "DomLifeGross",
                                       "ForLifeGross", "LtdRelDate", 
                                       "LtdOpenTh", "WRelDate", "WOpenTh", 
                                       "WidestTh", "Genre1", "Genre2", 
                                       "Genre3", "Genre4", "Awards"])
df = df.fillna(value=np.nan)
df[["Budget","DomLifeGross","ForLifeGross"]] = df[["Budget", "DomLifeGross", 
                                                    "ForLifeGross"]].astype(float)
#print df.head()
print df.dtypes
#print df.describe()
#print df.std()

#y, X = dmatrices('DomLifeGross ~ 
