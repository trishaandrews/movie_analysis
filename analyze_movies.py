import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns
from patsy import dmatrices
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

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
COUNTRYLIM = 10

sns.set(style="whitegrid", color_codes=True)

def initial_dataframe():
    movies = unpickle(PICKLEDIR + "cleanmovies.pkl")
    df = pd.DataFrame.from_items(movies.items(), orient='index', 
                               columns=["OriginC", "Budget", "DomLifeGross",
                                       "ForLifeGross", "LtdRelDate", 
                                       "LtdOpenTh", "WRelDate", "WOpenTh", 
                                       "WidestTh", "Genres", "Awards"])
    df["Genres"] = consolidate_genres(df)
    df = set_dataframe(df)
    pickle_stuff(PICKLEDIR + "df", df)
    return df

def set_dataframe(df):
    df = make_floats(df)
    df = make_floats(df)
    df_ = df[df["WidestTh"] >= THEATERLIM] 
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
    df["LtdRelDate"] = df["LtdRelDate"].map(lambda x: x.year)
    df["WRelDate"] = pd.to_datetime(df["WRelDate"], infer_datetime_format=True)
    df["WRelDate"] = df["WRelDate"].map(lambda x: x.year)
    return df

def separate_genres(df):
    s = pd.Series(df["Genres"])
    #if len(s) > 1:
    d_f = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    df = df.drop("Genres", 1)
    df = pd.concat([df, d_f], axis=1)
    return df

def make_model(X, y):
    model = sm.OLS(y,X)
    results = model.fit()
    print "Model R Squared:", results.rsquared
    print "Model Adj. R Squared:", results.rsquared_adj
    predicted = results.predict()
    return results, predicted

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

def domestic_origin_wopenth(df):
    y, X = dmatrices("DomLifeGross ~ OriginC + WOpenTh", data=df,
                     return_type = 'dataframe')
    print "Domestic from Country and Wide Opening Theaters"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_origin_wopenth_wreldate(df):
    y, X = dmatrices("DomLifeGross ~ OriginC + WOpenTh + WRelDate", data=df,
                     return_type = 'dataframe')
    print "Domestic from Country + Opening Theaters + Release Date"
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

def domestic_genre_by_country(df, country=None):
    '''Domestic Life Gross from Genre per Country'''
    if country:
        print country
        dfC = df.loc[df["OriginC"] == country]
    else:
        print "All Countries"
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
    print "Domestic from Genres"
    df_g = separate_genres(df)
    cols = list(df_g)
    domestic = df_g["DomLifeGross"]
    genres = df_g.iloc[:,10:len(cols)]
    genres.insert(0, "Intercept", 1)
    results, predicted = make_model(genres, domestic)
    return genres, domestic, results, predicted

def plot_domestic_genre(df, results):
    df_g = separate_genres(df)
    x= list(results.params.index)[2:] #no Intercept or 3D genre
    print x
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    for g in x:
        sns.regplot(x=g, y="DomLifeGross", data=df_g, color=data_c, line_kws = {"color" : pred_c})
    sns.plt.show()

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

def plot_domestic_foreign(X, y):
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: '%.0f'%(x*1e-6)))
    sns.regplot(x=X["ForLifeGross"], y=y["DomLifeGross"], color=data_c, line_kws = {"color" : pred_c})
    plt.xlabel("Foreign Lifetime Gross (millions)")
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.show()

def plot_domestic_budget(X, y):
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: '%.0f'%(x*1e-6)))
    sns.regplot(x=X["Budget"], y=y["DomLifeGross"], color=data_c, line_kws = {"color" : pred_c})
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
    sns.despine()
    sns.plt.show()

def plot_domestic_origin(df, predicted=None):
    ax = plt.subplot(111)
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    sns.stripplot(x="DomLifeGross", y="OriginC", data=df)
    sns.plt.xlabel("Domestic Lifetime Gross (millions)")
    sns.plt.ylabel("Country of Origin")
    sns.despine()
    sns.plt.show()

def plot_domestic_origin_pred(df, predicted):
    df_seaborn = pd.DataFrame(zip(df["DomLifeGross"], predicted), columns = ["ActDomLifeGross", "PredDomLifeGross"])
    print df_seaborn.head()
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    sns.regplot(x="ActDomLifeGross", y="PredDomLifeGross", data=df_seaborn, 
                color=data_c, line_kws = {"color" : pred_c})
    sns.plt.show()

def plot_domestic_origin_bar(df):
    grouped = df.groupby("OriginC")
    means = grouped.DomLifeGross.mean()
    errors = grouped.DomLifeGross.std()
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    means.plot(x="OriginC", yerr=errors, ax=ax, kind='bar', color=data_c, 
               ecolor=pred_c)
    grosslabel = "Domestic Lifetime Gross (millions)"
    clabel = "Country of Origin"
    ax.xaxis.grid(False)
    plt.xlabel(clabel)
    plt.ylabel(grosslabel)
    plt.subplots_adjust(bottom=0.35)
    plt.show()

def plot_domestic_wopenth(X, y, log=False):
    ax = plt.subplot(111)
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    sns.regplot(x=X["WOpenTh"], y=y["DomLifeGross"], color = data_c, line_kws = {"color" : pred_c})
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.xlabel("Wide Opening Theaters")
    plt.show()

def plot_domestic_widestth(X, y, log=False):
    ax = plt.subplot(111)
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, pos: '%.0f'%x))
    sns.regplot(x=X["WidestTh"], y=y["DomLifeGross"], color=data_c, 
                line_kws = {"color" : pred_c})
    plt.ylabel("Domestic Lifetime Gross (millions)")
    plt.xlabel("Theaters at Widest Release")
    plt.show()

def plot_domestic_genre_by_country(df, country=None):
    x, y, results, predicted = domestic_genre_by_country(df, country)
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
    plt.subplots_adjust(left=0.25)
    plt.show()
    return results

def plot_by_genre_mean(df, unknown = False):
    df_g = separate_genres(df)
    cols = list(df_g)
    gen_cols = cols[10:]
    xsk = []
    ysk = []
    errs = []
    for c in gen_cols:
        if unknown:
            xsk.append(c)
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
        plt.xlabel("Number of Movies")
        plt.ylabel("Genre")
        plt.subplots_adjust(left=0.35)
        plt.show()

def plot_genre_country_count(df, country):
    '''plot number of movies per genre per country'''
    dfC = df.loc[df["OriginC"] == country]
    dfC_g = separate_genres(dfC)
    cols = list(dfC_g)
    gen_cols = cols[10:]
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

def wreldate_hist(df):
    '''Plot a histogram of movies per year'''
    wd = df["WRelDate"]
    bins = df["WRelDate"].max() - df["WRelDate"].min()
    wd.hist(bins=bins, color=data_c, alpha=data_al)
    plt.xlabel("Release Year")
    plt.ylabel("# Movies")
    plt.title("Foreign Movies per Year")
    plt.show()

def challenge_1(df):
    X, y, results, predicted = domestic_constant(df)
    plot_domestic_constant(X, y, predicted)
    plot_residuals(results)

def challenge_2(df):
    X, y, results, predicted = domestic_foreign(df)
    plot_domestic_foreign(X,y)
    plot_residuals(results)

def challenge_3(df):
    X, y, results, predicted = domestic_origin(df)
    plot_domestic_origin(df, predicted)
    plot_residuals(results)

def challenge_4(df):
    X, y, results, predicted = domestic_origin_foreign(df)
    plot_residuals(results)

def run_all(df):
    X, y, results, predicted = domestic_foreign(df)
    plot_domestic_foreign(X, y)
    
    X, y, results, predicted = domestic_wopenth(df)
    plot_domestic_wopenth(X, y)
    
    X,y, results, predicted = domestic_wopenth(df, True)
    plot_domestic_wopenth(X, y, True)
        
    X, y, results, predicted = domestic_genre(df)
    X, y, results, predicted = domestic_foreign_wopenth(df)
    
    X, y, results, predicted = domestic_widestth(df)
    plot_domestic_widestth(X, y)
    X, y, results, predicted = domestic_widestth(df, True)
    plot_domestic_widestth(X, y, True)

    X, y, results, predicted = domestic_origin(df)
    plot_domestic_origin_pred(df, predicted) #this is bad/wrong(?)
    X, y, results, predicted = domestic_origin_foreign(df)
    print results.summary()
    plot_domestic_origin_bar(df)
    plot_by_genre_mean(df)

    X, y, results, predicted = domestic_budget(df)
    plot_domestic_budget(X, y)
    
    X, y, results, predicted = domestic_genre(df)
    plot_domestic_genre(df, results)
    plot_by_genre_count(df)
    plot_by_genre_count(df, True)
    
    country_list = ["INDIA", "FRANCE"]
    for country in country_list:
        plot_genre_country_count(df, country)
        results = plot_domestic_genre_by_country(df, country)
    plot_domestic_genre_by_country(df)

def run_challenges(df):
    challenge_1(df)
    challenge_2(df)
    challenge_3(df)
    challenge_4(df)

def run_specifics(df, country):
    X, y, results, predicted = domestic_foreign(df)
    plt.title(country)
    plot_domestic_foreign(X, y)
    
    X, y, results, predicted = domestic_wopenth(df)
    plt.title(country)
    plot_domestic_wopenth(X, y)
    
    #X,y, results, predicted = domestic_wopenth(df, True)
    #plt.title(country)
    #plot_domestic_wopenth(X, y, True)
            
    X, y, results, predicted = domestic_foreign_wopenth(df)
    
    X, y, results, predicted = domestic_widestth(df)
    plt.title(country)
    plot_domestic_widestth(X, y)
    #X, y, results, predicted = domestic_widestth(df, True)
    #plot_domestic_widestth(X, y, True)

    #plot_by_genre_mean(df)
    
    X, y, results, predicted = domestic_genre(df)
    #plot_domestic_genre(df, results)
    #plot_by_genre_count(df)
    #plot_by_genre_count(df, True)
    
def test_train(X, y, title="All Countries", xlab=None, ylab="Domestic (millions)", r_state=None, set_x=False, plot=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=r_state)
    
    smols = sm.OLS(y_train,X_train).fit()
    smols_pred_y = smols.predict(X_test)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    resids = []
    #residual = observed - predicted
    resids =  [y_test[i] - regr.predict(x[1])[1] for i, x in enumerate(X_test)]
    #for i, x_r in enumerate(X_test)
    #    resids.append(y_test[i] - regr.predict(x_r[1])[1])
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - 
                                                     y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('R Squared score: %.2f' % regr.score(X_test, y_test))

    if plot:
        plot_expected(X_test, y_test, regr, resids, title, xlab, ylab, set_x)

def test_split(df, country="All Countries", r_state=None):
    X, y, _, _ = domestic_foreign(df)
    test_train(X, y, title=country, xlab="Foreign", set_x=True, r_state=r_state)

    X, y, _, _ = domestic_wopenth(df)
    test_train(X, y, title=country, xlab="Wide Open Theaters", r_state=r_state)

    X, y, _, _ = domestic_widestth(df)
    test_train(X, y, title=country, xlab="Widest Theaters", r_state=r_state)

    X, y, _, _ = domestic_foreign_wopenth(df)
    test_train(X, y, plot=False)

def plot_expected(X_test, y_test, regr, resids, title="All Countries", 
                  xlab=None, ylab="Domestic", set_x=False):
    fig1 = plt.figure(1)
    ax=fig1.add_axes((.1,.3,.8,.6))
    X_list = map(lambda x: x[1], X_test)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    plt.scatter(X_list, y_test, color=data_c, alpha=data_al)
    plt.plot(X_list, regr.predict(X_test), color=pred_c, alpha=pred_al)
    plt.ylabel(ylab)
    plt.title(title)
    ax.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.grid(b=True, which='major')

    ax2=fig1.add_axes((.1,.1,.8,.2), sharex=ax)
    ax2.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.1f')%(x*1e-6)))
    plt.plot(X_list,resids,'ob')
    if set_x:
         ax2.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                    pos: ('%.0f')%(x*1e-6)))
    else:
        ax2.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                        pos: ('%.0f')%x))
    plt.grid(b=True, which='major')
    plt.xlabel(xlab)
    plt.show()

#initial_dataframe()
df = unpickle(PICKLEDIR + "df")

#X, y, results, predicted = domestic_wreldate(df)
#plot_domestic_wreldate(df)

#wreldate_hist(df)

#domestic_origin_wopenth(df)

#domestic_origin_wopenth_wreldate(df)

#run_challenges(df)
#run_all(df)
#X, y, results, predicted = domestic_wopenth(df)
#plot_domestic_wopenth(X, y)
#print "Dropping 'Crouching Tiger, Hidden Dragon'"
#df_sad = df[df["DomLifeGross"] < 80000000]    
#run_all(df_sad)

countries = df["OriginC"].values
counts = Counter(countries)
countries = counts.most_common(COUNTRYLIM)
countries = ["MEXICO"]
#[('INDIA', 267), ('FRANCE', 150), ('CHINA', 48), ('ITALY', 31), ('SPAIN', 30), ('MEXICO', 29), ('GERMANY', 28), ('SOUTH KOREA', 26), ('JAPAN', 19), ('HONG KONG', 19)]

for c in countries:
    print c[0]
    dfC = df.loc[df["OriginC"] == c]#[0]]
    run_specifics(dfC, c)#[0])
    test_split(dfC, c, r_state=2)#1,8,10 2is good for india, france
#print countries
countries_full = counts.most_common()
countries_full.sort()
print countries_full
xs = []
ys = []
for c in countries_full:
    xs.append(c[0])
    ys.append(c[1])
#sns.barplot(ys,xs, color=data_c, alpha=pred_al)
#plt.xlabel("Number of Movies")
#plt.ylabel("Country")
#plt.show()

print "Test Train for Action Genre"
df_g = separate_genres(df)
df_a = df_g[df_g["Action"] == 1]
#test_split(df_a, r_state=11)

print "Dropping 'Crouching Tiger, Hidden Dragon'"
df_sad = df[df["DomLifeGross"] < 80000000]
df_g = separate_genres(df_sad)
df_a = df_g[df_g["Action"] == 1]

ax = plt.subplot(111)
ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                               pos: ('%.0f')%(x*1e-6)))
sns.regplot(x="WOpenTh", y="DomLifeGross", data=df_a, color=data_c, line_kws = {"color" : pred_c})
plt.title("Action, no Crouching Tiger")
plt.xlabel("Wide Opening Theaters")
plt.ylabel("Domestic Lifetime Gross (millions)")
sns.plt.show()

test_split(df_a, r_state=2, country="Action")


cols = list(df_g)
genres = df_g.iloc[:,10:]
counts = genres.sum()
new_list = zip(list(genres), counts)
genres_counts = sorted(new_list, reverse=True, key=lambda x: x[1])
print genres_counts[:10]
#[('Unknown', 709.0), ('Action', 58.0), ('Gay / Lesbian', 14.0), ('Horror', 14.0), ('War', 12.0), ('Cannes Film Festival', 9.0), ('Drama', 9.0), ('Documentary', 8.0), ('Thriller', 8.0), ('3D', 7.0)]

# #movies/country hist/bar
