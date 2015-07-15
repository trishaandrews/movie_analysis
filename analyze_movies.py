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
data_al = 0.6
THEATERLIM = 10

sns.set(style="whitegrid", color_codes=True)

def initial_dataframe():
    movies = unpickle(PICKLEDIR + "cleanmovies.pkl")
    df = pd.DataFrame.from_items(movies.items(), orient='index', 
                               columns=["OriginC", "Budget", "DomLifeGross",
                                       "ForLifeGross", "LtdRelDate", 
                                       "LtdOpenTh", "WRelDate", "WOpenTh", 
                                       "WidestTh", "Genres", "Awards"])
    df["Genres"] = consolidate_genres(df)
    #print df.head()
    df = set_dataframe(df)
    #print df.describe()
    pickle_stuff(PICKLEDIR + "df", df)
    return df

def set_dataframe(df):
    df = make_floats(df)
    df = make_floats(df)
    df_ = df[df["WidestTh"] >= THEATERLIM] 
    #dfl = df_[df_["WOpenTh"] >= THEATERLIM]
    #df_g = separate_genres(dfl)
    df = make_datetime(df_)
    df["OriginC"] = df["OriginC"].str.upper()
    return df

def consolidate_genres(df):
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
    return newseries

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
    #if len(s) > 1:
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

def domestic_foreign_wopenth(df):
    y, X = dmatrices("DomLifeGross ~ ForLifeGross + WOpenTh", data=df, 
                     return_type = 'dataframe')
    print "Domestic from Foreign Gross + Wide Opening Theaters"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_budget(df):
    y, X = dmatrices("DomLifeGross ~ Budget", data=df, return_type='dataframe')
    print "Domestic from Budget"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_wopenth(df, log=False):
    y, X = dmatrices("DomLifeGross ~ WOpenTh", data=df, 
                     return_type = 'dataframe')
    print "Domestic from Wide Opening Theaters"
    if log:
        X = np.log(X)
        y = np.log(y)
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_widestth(df, log=False):
    y, X = dmatrices("DomLifeGross ~ WidestTh", data=df, 
                     return_type = 'dataframe')
    print "Domestic from Widest Release"
    if log:
        X = np.log(X)
        y = np.log(y)
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_wreldate(df):
    df_n = df.dropna(subset=["WRelDate"])
    y, X = dmatrices("DomLifeGross ~ WRelDate", data=df_n, 
                     return_type = 'dataframe')
    #print X.head()
    print "Domestic from Wide Release Date"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def by_country(df, country=None):
    if country:
        dfC = df.loc[df["OriginC"] == country]
    else:
        dfC = df
    dfC_g = separate_genres(dfC)
    cols = list(dfC_g)
    domestic = dfC_g["DomLifeGross"]
    genres = dfC_g.iloc[:,10:len(cols)]
    genres.insert(0, "Intercept", 1)
    results, predicted = make_model(genres, domestic)
    #print results.summary()
    #print results.params
    return genres, domestic, results, predicted

def domestic_genre(df):
    df_g = separate_genres(df)
    cols = list(df_g)
    domestic = df_g["DomLifeGross"]
    genres = df_g.iloc[:,10:len(cols)]
    genres.insert(0, "Intercept", 1)
    results, predicted = make_model(genres, domestic)
    #print results.summary()
    #print results.params
    return genres, domestic, results, predicted

#def genres_df

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

def plot_domestic_budget(X, y, predicted):
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: '%.0f'%(x*1e-6)))
    plt.scatter(x=X["Budget"], y=y["DomLifeGross"], color=data_c, alpha=data_al)
    plt.subplot(111)
    plt.plot(X["Budget"], predicted, color=pred_c, alpha=pred_al)
    plt.xlabel("Budget (millions)")
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.show()

def plot_domestic_wreldate(df):
    np.random.seed(sum(map(ord, "categorical")))
    df_ = df.sort(["WRelDate"])
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    g = sns.stripplot(x="WRelDate", y="DomLifeGross", data=df_)
    sns.plt.ylabel("Domestic Lifetime Gross (millions)")
    sns.plt.xlabel("Country of Origin")
    plt.xticks(rotation=90)
    #sns.plt.margins(0.2)
    sns.despine()
    sns.plt.show()

def plot_domestic_origin(df):
    np.random.seed(sum(map(ord, "categorical")))
    ax = plt.subplot(111)
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    #ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    g = sns.stripplot(x="DomLifeGross", y="OriginC", data=df)
    sns.plt.xlabel("Domestic Lifetime Gross (millions)")
    sns.plt.ylabel("Country of Origin")
    #print "domestic origin"
    #plt.xticks(rotation=90)
    #sns.plt.margins(0.2)
    sns.despine()
    sns.plt.show()

def plot_domestic_origin_bar(df, kind='bar'):
    grouped = df.groupby("OriginC")
    means = grouped.DomLifeGross.mean()
    errors = grouped.DomLifeGross.std()
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    #ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    means.plot(x="OriginC", yerr=errors, ax=ax, kind=kind, color=data_c, ecolor=pred_c)
    #plt.yticks(df["OriginC"])
    grosslabel = "Domestic Lifetime Gross (millions)"
    clabel = "Country of Origin"
    #if kind=='barh':
    #    plt.ylabel(clabel)
    #    plt.xlabel(grosslabel)
    ax.xaxis.grid(False)
    plt.xlabel(clabel)
    plt.ylabel(grosslabel)
    plt.subplots_adjust(bottom=0.35)
    plt.show()

def plot_domestic_wopenth(X, y, predicted, log=False):
    ax = plt.subplot(111)
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    plt.scatter(x=X["WOpenTh"], y=y["DomLifeGross"], color = data_c, 
                alpha=data_al)
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.xlabel("Wide Opening Theaters")
    plt.subplot(111)
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    plt.scatter(x=X["WOpenTh"], y=predicted, color=pred_c, alpha=pred_al)
    plt.show()

def plot_domestic_widestth(X, y, predicted, log=False):
    ax = plt.subplot(111)
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    plt.scatter(x=X["WidestTh"], y=y["DomLifeGross"], color = data_c, 
                alpha=data_al)
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.xlabel("Theaters at Widest Release")
    plt.subplot(111)
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    plt.scatter(x=X["WidestTh"], y=predicted, color=pred_c, alpha=pred_al)
    plt.show()

def plot_by_country(df, country=None):
    x, y, results, predicted = by_country(df, country)
    xs = list(results.params.index)[1:]
    ys = list(results.params.values)[1:]
    sns.barplot(ys,xs, color=data_c, alpha=pred_al)
    ax = sns.plt.subplot(111)
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                               pos: ('%.0f')%(x*1e-6)))
    sns.plt.xlabel("Domestic Lifetime Gross (millions)")
    if country is None:
        country = "All Countries"
    plt.title("Params, " + country)
    #plt.xticks(rotation=90)
    plt.subplots_adjust(left=0.25)
    plt.show()
    return results

def plot_by_genre_mean(df, unknown = False):
    df_g = separate_genres(df)
    cols = list(df_g)
    gen_cols = cols[10:]
    domestic = df_g["DomLifeGross"]
    xsk = []
    ysk = []
    errs = []
    for c in gen_cols:
        if unknown:
            xsk.append(c)
            #ysk.append(df_g[c].sum())
            df1g = df.loc[df_g[c] > 0 ]
            ysk.append(df1g["DomLifeGross"].mean())
            errs.append(df1g["DomLifeGross"].std())
        else:
            if c != "Unknown":
                xsk.append(c)
                df1g = df.loc[df_g[c] > 0 ]
                ysk.append(df1g["DomLifeGross"].mean())
                errs.append(df1g["DomLifeGross"].std())
    if len(xsk) > 0:
        ax = plt.subplot(111)
        ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
        sns.barplot(x=xsk, y=ysk, yerr=errs, color=data_c, ecolor=pred_c)
        plt.xticks(rotation=90)
        plt.xlabel("Genre")
        plt.ylabel("Domestic Lifetime Gross")
        plt.subplots_adjust(bottom=0.45)
        plt.show()

def plot_by_genre_count(df, unknown=False):
    df_g = separate_genres(df)
    cols = list(df_g)
    gen_cols = cols[10:]
    domestic = df_g["DomLifeGross"]
    xsk = []
    ysk = []
    ax = plt.subplot(111)
    for c in gen_cols:
        if unknown:
            xsk.append(c)
            ysk.append(df_g[c].sum())
            plt.title("# Movies by Genre")
        else:
            if c != "Unknown":
                xsk.append(c)
                ysk.append(df_g[c].sum())
                plt.title("# Movies by Known Genre")
    if len(xsk) > 0:
        sns.barplot(x=ysk, y=xsk, color=data_c, alpha=pred_al)
        #plt.xticks(rotation=90)
        plt.xlabel("Number of Movies")
        plt.ylabel("Genre")
        plt.subplots_adjust(left=0.35)
        plt.show()

def plot_by_country_count(df, country):
    dfC = df.loc[df["OriginC"] == country]
    dfC_g = separate_genres(dfC)
    cols = list(dfC_g)
    gen_cols = cols[10:]
    domestic = dfC_g["DomLifeGross"]
    xsk = []
    ysk = []
    for c in gen_cols:
        if c != "Unknown":
            xsk.append(c)
            ysk.append(dfC_g[c].sum())
    if len(xsk) > 0:
        ax1 = plt.subplot(111)
        sns.barplot(x=ysk, y=xsk, color=data_c, alpha=pred_al)
        plt.xlabel("Number of Movies")
        plt.ylabel("Genre")
        plt.title(country + ", Movies of Known Genre")
        plt.subplots_adjust(left=0.25)
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


def run_all(df):
    X, y, results, predicted = domestic_foreign(df)
    plot_domestic_foreign(X, y, predicted)
    
    X, y, results, predicted = domestic_wopenth(df)
    plot_domestic_wopenth(X, y, predicted)
    #print results.summary()
    X,y, results, predicted = domestic_wopenth(df, True)
    plot_domestic_wopenth(X, y, predicted, True)
    #print results.summary()
    
    X, y, results, predicted = domestic_foreign_wopenth(df)
    print results.summary()

    X, y, results, predicted = domestic_widestth(df)
    plot_domestic_widestth(X, y, predicted)
    X, y, results, predicted = domestic_widestth(df, True)
    plot_domestic_widestth(X, y, predicted, True)

    X, y, results, predicted = domestic_origin(df)
    #print results.summary()
    plot_domestic_origin(df)
    X, y, results, predicted = domestic_origin_foreign(df)
    #print results.summary()
    plot_domestic_origin_bar(df)
    plot_by_genre_mean(df)

    X, y, results, predicted = domestic_budget(df)
    #print results.summary()
    plot_domestic_budget(X, y, predicted)
    
    X, y, results, predicted = domestic_genre(df)
    #print results.summary()
    plot_by_genre_count(df)
    plot_by_genre_count(df, True)
    
    
    country_list = ["INDIA", "FRANCE"]#,  "BRAZIL", "ARGENTINA", "BELGIUM", "DENMARK", "GERMANY", "CANADA", "HOLLAND", "ISRAEL", "JAPAN", "LEBANON", "NORWAY", "PALESTINE", "ROMANIA", "RUSSIA", "SWITZERLAND", "SOUTH KOREA", "CZECH", "SWEDEN", "THAILAND", "TAIWAN", "POLAND", "INDONESIA", "NETHERLANDS"]#
    #France, india
    for country in country_list:
        plot_by_country_count(df, country)
        results = plot_by_country(df, country)
        #    #print results.summary()
    plot_by_country(df)


#initial_dataframe()
df = unpickle(PICKLEDIR + "df")
#df = set_dataframe(df)
#print df.head()
#print df.dtypes

#domestic_constant(df)
X, y, results, predicted = domestic_wreldate(df)
#print results.rsquared
plot_domestic_wreldate(df)

#does country or genre matter more? reduce # genres
#drop movies with lim release of less than 5-10
#linear in logvslog

#crunchbase.com, angel.co


run_all(df)

print "Dropping 'Crouching Tiger, Hidden Dragon'"
df_sad = df[df["DomLifeGross"] < 80000000]    
run_all(df_sad)

#scikit learn crossvalidation train_test_split
