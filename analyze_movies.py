import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from patsy import dmatrices

PICKLEDIR = "./pkls/"

def pickle_stuff(filename, data):
    with open(filename, 'w') as picklefile:
        pickle.dump(data, picklefile)

def unpickle(filename):
    with open(filename, 'r') as picklefile:
        old_data = pickle.load(picklefile)
    return old_data

def ols_domestic_constant(df):
    y, X = dmatrices('DomLifeGross ~ 1', data=df, return_type = 'dataframe')
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def make_model(X, y):
    #print y.head(), X.head()
    model = sm.OLS(y,X)
    results = model.fit()
    #print results.summary()
    #print "Parameters:", results.params
    print "R Squared", results.rsquared
    #print "Standard Errors", results.bse
    predicted = results.predict()
    #print "Predicted Values", predicted
    return results, predicted

def plot_domestic_constant(X, y, predicted):
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))

    plt.scatter(X["Intercept"], y, alpha=0.3)
    plt.scatter(X["Intercept"], predicted, color="red", alpha=0.6)
    plt.xlabel("Ones")
    plt.ylabel("Domestic Lifetime Gross")
    plt.show()

def plot_residuals(results):
    ax = plt.subplot(111)
    plt.hist(results.resid, bins=75, alpha=0.5)
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: ('%.0f')%(x*1e-6)))
    plt.ylabel("Count")
    plt.xlabel("Residual in Millions")
    plt.show()

def plot_domlifegross_forlifegross(X,y, predicted):
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    
    plt.scatter(x=X["ForLifeGross"], y=y["DomLifeGross"], alpha=0.3)
    plt.subplot(111)
    plt.plot(X["ForLifeGross"], predicted, color="red", alpha=0.6)
    plt.xlabel("Foreign Lifetime Gross")
    plt.ylabel("Domestic Lifetime Gross")
    plt.show()

def plot_domestic_origin(X, y, predicted):
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    plt.scatter(x=X, y=y, alpha=0.3)
    plt.show()
    

def challenge_1(df):
    X, y, results, predicted = ols_domestic_constant(df)
    plot_domestic_constant(X, y, predicted)
    plot_residuals(results)

def challenge_2(df):
    y, X = dmatrices('DomLifeGross ~ ForLifeGross', 
                     data=df, return_type = 'dataframe')
    results, predicted = make_model(X, y)
    plot_domlifegross_forlifegross(X,y, predicted)
    plot_residuals(results)


movies = unpickle(PICKLEDIR + "cleanmovies.pkl")

df = pd.DataFrame.from_items(movies.items(), orient='index', 
                               columns=["OriginC", "Budget", "DomLifeGross",
                                       "ForLifeGross", "LtdRelDate", 
                                       "LtdOpenTh", "WRelDate", "WOpenTh", 
                                       "WidestTh", "Genres", "Awards"])
#1", "Genres2", "Genres3", "Genres4"
df = df.fillna(value=np.nan)
df[["Budget","DomLifeGross","ForLifeGross", "LtdOpenTh", "WOpenTh", 
    "WidestTh"]] = df[["Budget", "DomLifeGross", "ForLifeGross", "LtdOpenTh", 
                       "WOpenTh", "WidestTh"]].astype(float)

s = pd.Series(df["Genres"])
d_f = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
#print d_f.head()
#print df.head()
df = df.drop("Genres", 1)
#df = df.append(d_f)#, overwrite=False)#, filter_func=None, raise_conflict=False)
#print df.head()
#print d_f.head()
#print df.dtypes
#print df.head
#print df.describe()


challenge_1(df)
challenge_2(df)

def challenge_3(df):
    y, X = dmatrices('DomLifeGross ~ OriginC', data=df, return_type = 'dataframe')
    results, predicted = make_model(X, y)
    #plot_domestic_origin(X,y, predicted)
    plot_residuals(results)

    y, X = dmatrices("DomLifeGross ~ OriginC + ForLifeGross", data=df, 
                     return_type = 'dataframe')
    results, predicted = make_model(X, y)
    plot_residuals(results)


challenge_3(df)
