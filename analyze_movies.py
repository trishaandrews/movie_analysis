import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns
from patsy import dmatrices

'''
columns=["OriginC", "Budget", "DomLifeGross","ForLifeGross", "LtdRelDate", 
          "LtdOpenTh", "WRelDate", "WOpenTh", "WidestTh", "Genres", "Awards"]
'''

PICKLEDIR = "./pkls/"

pred_c = "orange"
pred_al = 0.6
data_c = "purple"
data_al = 0.3
THEATERLIM = 5

sns.set(style="whitegrid", color_codes=True)

def pickle_stuff(filename, data):
    with open(filename, 'w') as picklefile:
        pickle.dump(data, picklefile)

def unpickle(filename):
    with open(filename, 'r') as picklefile:
        old_data = pickle.load(picklefile)
    return old_data

def make_floats(df):
    df = df.fillna(value=np.nan)
    df[["Budget","DomLifeGross","ForLifeGross", "LtdOpenTh", "WOpenTh", 
        "WidestTh"]] = df[["Budget", "DomLifeGross", "ForLifeGross", 
                           "LtdOpenTh", "WOpenTh", "WidestTh"]].astype(float)
    return df

def make_datetime(df):
    df["LtdRelDate"] = pd.to_datetime(df["LtdRelDate"], 
                                      infer_datetime_format=True)
    df["WRelDate"] = pd.to_datetime(df["WRelDate"], infer_datetime_format=True)
    return df

def separate_genres(df):
    s = pd.Series(df["Genres"])
    d_f = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    df = df.drop("Genres", 1)
    df = pd.concat([df, d_f], axis=1)
    return df

def domestic_constant(df):
    y, X = dmatrices('DomLifeGross ~ 1', data=df, return_type = 'dataframe')
    print "Domestic from 1s"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_foreign(df):
    y, X = dmatrices('DomLifeGross ~ ForLifeGross', 
                     data=df, return_type = 'dataframe')
    print "Domestic from Foreign"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_origin(df):
    y, X = dmatrices('DomLifeGross ~ OriginC', data=df, 
                     return_type = 'dataframe')
    print "Domestic from Origin Country"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_origin_foreign(df):
    y, X = dmatrices("DomLifeGross ~ OriginC + ForLifeGross", data=df, 
                     return_type = 'dataframe')
    print "Domestic from Origin Country + Foreign"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_wopenth(df):
    y, X = dmatrices("DomLifeGross ~ WOpenTh", data=df, 
                     return_type = 'dataframe')
    print "Domestic from Wide Opening Theaters"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_widestth(df):
    y, X = dmatrices("DomLifeGross ~ WidestTh", data=df, 
                     return_type = 'dataframe')
    print "Domestic from Widest Release"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_wreldate(df):
    df_n = df.dropna(subset=["WRelDate"])
    y, X = dmatrices("DomLifeGross ~ WRelDate", data=df_n, 
                     return_type = 'dataframe')
    print X.head()
    print "Domestic from Wide Release Date"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def by_country(df, country):
    dfC = df.loc[df["OriginC"] == country]
    dfC_g = separate_genres(dfC)
    cols = list(dfC_g)
    domestic = dfC_g["DomLifeGross"]
    genres = dfC_g.iloc[:,10:len(cols)]
    genres.insert(0, "Intercept", 1)
    results, predicted = make_model(genres, domestic)
    print results.summary()
    print results.params
    #print dir(results)
    return genres, domestic, results, predicted

def make_model(X, y):
    #print y.head(), X.head()
    model = sm.OLS(y,X)
    results = model.fit()
    #print results.summary()
    #print "Parameters:", results.params
    print "R Squared:", results.rsquared
    print "Adj. R Squared:", results.rsquared_adj
    #print "Standard Errors", results.bse
    predicted = results.predict()
    #print "Predicted Values", predicted
    return results, predicted

def plot_domestic_constant(X, y, predicted):
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    plt.scatter(X["Intercept"], y, color=data_c, alpha=data_al)
    plt.scatter(X["Intercept"], predicted, color=pred_c, alpha=pred_al)
    plt.xlabel("Ones")
    plt.ylabel("Domestic Lifetime Gross")
    plt.show()

def plot_residuals(results):
    ax = plt.subplot(111)
    plt.hist(results.resid, bins=75, alpha=0.5, color=data_c)
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    plt.ylabel("Count")
    plt.xlabel("Residual in Millions")
    plt.show()

def plot_domestic_foreign(X,y, predicted):
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: '%.0f'%(x*1e-6)))
    plt.scatter(x=X["ForLifeGross"], y=y["DomLifeGross"], color=data_c, alpha=data_al)
    plt.subplot(111)
    plt.plot(X["ForLifeGross"], predicted, color=pred_c, alpha=pred_al)
    plt.xlabel("Foreign Lifetime Gross (millions)")
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.show()


def plot_domestic_origin(df):
    np.random.seed(sum(map(ord, "categorical")))
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    g = sns.stripplot(x="OriginC", y="DomLifeGross", data=df)
    sns.plt.ylabel("Domestic Lifetime Gross (millions)")
    sns.plt.xlabel("Country of Origin")
    plt.xticks(rotation=30)
    sns.plt.margins(0.2)
    sns.despine()
    #plt.scatter(x=X, y=y, alpha=0.3)
    sns.plt.show()

def plot_domestic_country_genre(df):
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    #plt.(, y=y["DomLifeGross"], color = data_c, 
    #            alpha=data_al)
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.xlabel("Wide Opening Theaters")
    plt.subplot(111)
    plt.plot(X["WOpenTh"], predicted, color=pred_c, alpha=pred_al)
    plt.show()

def plot_domestic_wopenth(X, y, predicted):
    ax = plt.subplot(111)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    plt.scatter(x=X["WOpenTh"], y=y["DomLifeGross"], color = data_c, 
                alpha=data_al)
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.xlabel("Wide Opening Theaters")
    plt.subplot(111)
    ax.set_yscale('log')
    plt.scatter(x=X["WOpenTh"], y=predicted, color=pred_c, alpha=pred_al)
    plt.show()

def plot_domestic_widestth(X, y, predicted):
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    plt.scatter(x=X["WidestTh"], y=y["DomLifeGross"], color = data_c, 
                alpha=data_al)
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.xlabel("Theaters at Widest Release")
    plt.subplot(111)
    plt.plot(X["WidestTh"], predicted, color=pred_c, alpha=pred_al)
    plt.show()

def plot_domestic_wreldate(df):
    np.random.seed(sum(map(ord, "categorical")))
    
    ax = sns.plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    sns.stripplot(x="WRelDate", y="DomLifeGross", data=df)
    sns.plt.ylabel("Domestic Lifetime Gross (millions)")
    sns.plt.xlabel("Wide Release Date")
    sns.despine()
    #g.set_xticklabels(rotation=30)
    sns.plt.margins(0.2)
    #sns.plt.subplot(111)
    #sns.plt.plot(x="WRelDate", predicted, color=pred_c, alpha=pred_al)
    #np.random.seed(sum(map(ord, "regression")))
    #sns.lmplot(x="WRelDate", y="DomLifeGross", data=df);
    sns.plt.show()
    '''
    
    #print X
    plt.scatter(x=X["WRelDate"].to_pydatetime(), y=y["DomLifeGross"], color = data_c, 
                alpha=data_al)
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.xlabel("Wide Release Date")
    plt.subplot(111)
    plt.plot(X["WRelDate"], predicted, color=pred_c, alpha=pred_al)
    plt.show()
    '''

def plot_domestic_genre(df):
    np.random.seed(sum(map(ord, "categorical")))
    
    ax = sns.plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    sns.scatter(x="G", y="DomLifeGross", data=df)
    sns.plt.ylabel("Domestic Lifetime Gross (millions)")
    sns.plt.xlabel("Wide Release Date")
    sns.despine()
    plt.set_xticklabels(rotation=30)
    sns.plt.margins(0.2)
    sns.plt.show()

def plot_by_country(df):
    x, y, results, predicted = by_country(df, "FRANCE")
    xs = list(results.params.index)[1:]
    ys = list(results.params.values)[1:]
    sns.barplot(xs,ys)
    ax = sns.plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                               pos: ('%.0f')%(x*1e-6)))
    sns.plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.xticks(rotation=80)
    plt.subplots_adjust(bottom=0.45)
    plt.show()

def challenge_1(df):
    X, y, results, predicted = domestic_constant(df)
    plot_domestic_constant(X, y, predicted)
    plot_residuals(results)

def challenge_2(df):
    X, y, results, predicted = domestic_foreign(df)
    plot_domestic_foreign(X,y, predicted)
    plot_residuals(results)

def challenge_3(df):
    X, y, results, predicted = domestic_origin(df)
    plot_residuals(results)

def challenge_3_2(df):
    X, y, results, predicted = domestic_origin_foreign(df)
    plot_residuals(results)

movies = unpickle(PICKLEDIR + "cleanmovies.pkl")

df = pd.DataFrame.from_items(movies.items(), orient='index', 
                               columns=["OriginC", "Budget", "DomLifeGross",
                                       "ForLifeGross", "LtdRelDate", 
                                       "LtdOpenTh", "WRelDate", "WOpenTh", 
                                       "WidestTh", "Genres", "Awards"])
def consolidate_genres():
    newseries = df["Genres"]
    for genre in df["Genres"].iteritems():
        key = genre[0]
        newlist = []
        genlist = genre[1]
        if "Foreign Language" in genlist:
            genlist.remove("Foreign Language")
        if len(genlist) > 0:
            if len(genlist) > 1:
                if "Foreign" in genlist:
                    genlist.remove("Foreign")
                if "Unknown" in genlist:
                    genlist.remove("Unknown")
            for g in genlist:
                sep = g.split(" / ")
                if len(sep) > 1:
                    if sep[0] == "Foreign":
                        g = sep[1]
                    else:
                        g = " / ".join(sep)
                sep = g.split(" - ")
                if len(sep) > 1:
                    g = sep[0]
                if g == "Foreign":
                    g = "Unknown"
                newlist.append(g)
        newset = set(newlist)
        newlist = list(newset)
        newseries[key] = newlist
        
        #print genlist, newlist
    return newseries

#df["Genres"] = consolidate_genres()
#print df.head()
#df = make_floats(df)
#dfl = df[df["WOpenTh"] > THEATERLIM]
#df_g = separate_genres(dfl)
#df = make_datetime(dfl)
#df["OriginC"] = df["OriginC"].str.upper()
#print df.describe()

#pickle_stuff(PICKLEDIR + "dfl", df)

df = unpickle(PICKLEDIR + "df")
#print df.head()
#print df.dtypes

#domestic_constant(df)
#domestic_foreign(df)
#challenge_2(df)
#plot_domestic_origin(df)
#X, y, results, predicted = domestic_wopenth(dfl)
#plot_domestic_wopenth(X, y, predicted)

#X, y, results, predicted = domestic_widestth(df)
#plot_domestic_widestth(X, y, predicted)

#X, y, results, predicted = domestic_wreldate(df)
#plot_domestic_wreldate(df)

'''
X, y, results, predicted = by_country(df, "CHINA")
ax = plt.subplot(111)
ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                               pos: ('%.0f')%(x*1e-6)))
ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
plt.plot(results.params, color = data_c, alpha=data_al)
plt.plot(predicted)
plt.ylabel("Domestic Lifetime Gross (millions)")
plt.xlabel("Genres")
plt.subplot(111)
plt.plot(X["WOpenTh"], predicted, color=pred_c, alpha=pred_al)
plt.show()
'''
#does country or genre matter more? reduce # genres
#drop movies with lim release of less than 5-10
#linear in logvslog



