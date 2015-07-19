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
Order of values in pickled dictionary
columns=["OriginC", "Budget", "DomLifeGross","ForLifeGross", "LtdRelDate", 
          "LtdOpenTh", "WRelDate", "WOpenTh", "WidestTh", "Genres", "Awards"]
'''

#Initial variables
PICKLEDIR = "./pkls/"
THEATERLIM = 10 # movie must have been in at least this many theaters
COUNTRYLIM = 10 # number of top countries to include
pred_c = "orange"
pred_al = 0.6
data_c = "purple"
data_al = 0.6

sns.set(style="whitegrid", color_codes=True)

def pickle_stuff(filename, data):
    ''' open file '''
    with open(filename, 'w') as picklefile:
        pickle.dump(data, picklefile)

def unpickle(filename):
    ''' save file '''
    with open(filename, 'r') as picklefile:
        old_data = pickle.load(picklefile)
    return old_data

def initial_dataframe():
    '''create dataframe from pickled dictionary of movies and values'''
    movies = unpickle(PICKLEDIR + "cleanmovies.pkl")
    df = pd.DataFrame.from_items(movies.items(), orient='index', 
                               columns=["OriginC", "Budget", "DomLifeGross",
                                       "ForLifeGross", "LtdRelDate", 
                                       "LtdOpenTh", "WRelDate", "WOpenTh", 
                                       "WidestTh", "Genres", "Awards"])
    df = set_dataframe(df)
    pickle_stuff(PICKLEDIR + "df", df)
    return df

def set_dataframe(df):
    '''perform cleaning and data type setting operations on the dataframe'''
    df["Genres"] = consolidate_genres(df)
    df = make_floats(df)
    df = make_floats(df)
    df_ = df[df["WidestTh"] >= THEATERLIM] 
    df = make_datetime(df_)
    df["OriginC"] = df["OriginC"].str.upper()
    return df

def consolidate_genres(df):
    '''combine genres such as action - martial arts and action - wire fu as
    action. convert genres such as foreign / horror to just horror, but keep 
    genres such as gay / lesbian. remove foreign genre from movies with 
    additional genres. Otherwise, convert to unknown since all input movies 
    are foreign. This throws a warning, but ignore it'''
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
        newset = set(newlist) #remove duplicates
        newlist = list(newset) #but I want it to be a list
        newseries[key] = newlist
    return newseries

def make_floats(df):
    ''' convert columns of numbers to float, since they start as strings '''
    df = df.fillna(value=np.nan)
    df[["Budget","DomLifeGross","ForLifeGross", "LtdOpenTh", "WOpenTh", 
        "WidestTh"]] = df[["Budget", "DomLifeGross", "ForLifeGross", 
                           "LtdOpenTh", "WOpenTh", "WidestTh"]].astype(float)
    return df

def make_datetime(df):
    ''' convert applicable columns to year values from full dates/strings '''
    df["LtdRelDate"] = pd.to_datetime(df["LtdRelDate"], 
                                      infer_datetime_format=True)
    df["LtdRelDate"] = df["LtdRelDate"].map(lambda x: x.year)
    df["WRelDate"] = pd.to_datetime(df["WRelDate"], infer_datetime_format=True)
    df["WRelDate"] = df["WRelDate"].map(lambda x: x.year)
    return df

def separate_genres(df):
    '''since genres is a column of lists, convert to dummy variables'''
    s = pd.Series(df["Genres"])
    d_f = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    df = df.drop("Genres", 1)
    df = pd.concat([df, d_f], axis=1)
    return df

def make_model(X, y):
    '''model data with an ordinary least squares regression and prit R and R^2
    values'''
    model = sm.OLS(y,X)
    results = model.fit()
    print "Model R Squared:", results.rsquared
    print "Model Adj. R Squared:", results.rsquared_adj
    predicted = results.predict()
    return results, predicted

def domestic_foreign(df, plot=False):
    '''creates and can plot a model of domestic lifetime gross vs foreign 
    lifetime gross'''
    y, X = dmatrices('DomLifeGross ~ ForLifeGross', 
                     data=df, return_type = 'dataframe')
    print "Domestic from Foreign"
    results, predicted = make_model(X, y)
    if plot:
        plot_domestic_foreign(X, y)
    return X, y, results, predicted

def domestic_origin(df, plot=False):
    '''creates and can plot a model of domestic lifetime gross vs country of 
    origin'''
    y, X = dmatrices('DomLifeGross ~ OriginC', data=df, 
                     return_type = 'dataframe')
    print "Domestic from Origin Country"
    results, predicted = make_model(X, y)
    if plot:
        plot_domestic_origin_bar(df)
    return X, y, results, predicted
 
def domestic_origin_foreign(df):
    '''creates a model for domestic lifetime gross vs country of origin and
    foreign lifetime gross'''
    y, X = dmatrices("DomLifeGross ~ OriginC + ForLifeGross", data=df, 
                     return_type = 'dataframe')
    print "Domestic from Origin Country + Foreign"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_foreign_wopenth(df):
    '''creates a model for domestic lifetime gross vs foreign lifetime gross and
    number of theaters at widest opening'''
    y, X = dmatrices("DomLifeGross ~ ForLifeGross + WOpenTh", data=df, 
                     return_type = 'dataframe')
    print "Domestic from Foreign Gross + Wide Opening Theaters"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_origin_wopenth(df):
    '''creates a model for domestic lifetime gross vs country of origin and
    number of theaters at widest opening'''
    y, X = dmatrices("DomLifeGross ~ OriginC + WOpenTh", data=df,
                     return_type = 'dataframe')
    print "Domestic from Country + Wide Opening Theaters"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_origin_wopenth_wreldate(df):
    '''creates a model for domestic lifetime gross vs number of theaters at 
    widest opening and release/wide release date'''
    y, X = dmatrices("DomLifeGross ~ OriginC + WOpenTh + WRelDate", data=df,
                     return_type = 'dataframe')
    print "Domestic from Country + Opening Theaters + Release Date"
    results, predicted = make_model(X, y)
    return X, y, results, predicted

def domestic_budget(df, plot=False):
    '''creates a model for domestic lifetime gross vs budget'''
    y, X = dmatrices("DomLifeGross ~ Budget", data=df, return_type='dataframe')
    print "Domestic from Budget"
    results, predicted = make_model(X, y)
    if plot:
        plot_domestic_budget(X, y)
    return X, y, results, predicted

def domestic_wopenth(df, plot=False):
    '''creates a model for domestic lifetime gross vs number of theaters at 
    widest opening'''
    y, X = dmatrices("DomLifeGross ~ WOpenTh", data=df, 
                     return_type = 'dataframe')
    print "Domestic from Wide Opening Theaters"
    results, predicted = make_model(X, y)
    if plot:
        plot_domestic_wopenth(X, y)
    return X, y, results, predicted

def domestic_widestth(df, plot=False):
    '''creates a model for domestic lifetime gross vs largest number of 
    theaters shown in'''
    y, X = dmatrices("DomLifeGross ~ WidestTh", data=df, 
                     return_type = 'dataframe')
    print "Domestic from Widest Release"
    results, predicted = make_model(X, y)
    if plot:
        plot_domestic_widestth(X,y)
    return X, y, results, predicted

def domestic_wreldate(df, plot=False):
    '''creates a model for domestic lifetime gross vs widest release date'''
    df_n = df.dropna(subset=["WRelDate"])
    y, X = dmatrices("DomLifeGross ~ WRelDate", data=df_n, 
                     return_type = 'dataframe')
    print "Domestic from Wide Release Date"
    results, predicted = make_model(X, y)
    if plot:
        plot_domestic_wreldate(df)
    return X, y, results, predicted

def domestic_genre_by_country(df, country=None, plot=False):
    '''creates a model from domestic lifetime gross from genre per country'''
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
    if plot:
        plot_domestic_genre_by_country(df, country)
    return genres, domestic, results, predicted

def domestic_genre(df):
    '''creates a model of domestic lifetime gross from genre'''
    print "Domestic from Genres"
    df_g = separate_genres(df)
    cols = list(df_g)
    domestic = df_g["DomLifeGross"]
    genres = df_g.iloc[:,10:len(cols)]
    genres.insert(0, "Intercept", 1)
    results, predicted = make_model(genres, domestic)
    return genres, domestic, results, predicted

def movies_per_country(df):
    '''plots the number of movies per country'''
    countries = df["OriginC"].values
    counts = Counter(countries)
    countries_full = counts.most_common()
    countries_full.sort()
    xs = []
    ys = []
    for c in countries_full:
        xs.append(c[0])
        ys.append(c[1])
    sns.barplot(ys,xs, color=data_c, alpha=pred_al)
    plt.xlabel("Number of Movies")
    plt.ylabel("Country")
    plt.show()

def plot_residuals(results):
    '''plots the residuals from statsmodels regression model results'''
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
    '''plot number of movies per genre, toggle to include unknown genre'''
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
    
def plot_train_test_action(df):
    print "Test Train for Action Genre"    
    print "Dropping 'Crouching Tiger, Hidden Dragon'"
    df_sad = df[df["DomLifeGross"] < 80000000]
    df_g = separate_genres(df_sad)
    df_a = df_g[df_g["Action"] == 1]
    
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, 
                                                   pos: ('%.0f')%(x*1e-6)))
    sns.regplot(x="WOpenTh", y="DomLifeGross", data=df_a, color=data_c, 
                line_kws = {"color" : pred_c})
    plt.title("Action, no Crouching Tiger")
    plt.xlabel("Wide Opening Theaters")
    plt.ylabel("Domestic Lifetime Gross (millions)")
    sns.plt.show()
    test_split(df_a, r_state=2, country="Action")

def test_train(X, y, title="All Countries", xlab=None, 
               ylab="Domestic (millions)", r_state=None, set_x=False, 
               plot=True):
    '''creates a model from given X and y and performs test/train split and 
    regression. can plo the results'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                        random_state=r_state)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    resids = []
    #residual = observed - predicted
    resids =  [y_test[i] - regr.predict(x[1])[1] for i, x in enumerate(X_test)]
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
    '''perform train-test split on a series of models'''
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
    '''plot a scatter graph of test data under a train data prediction line, 
    with the difference between actual and predicted graphed below'''
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

def run_specifics(df, country):
    plt.title(country)
    X, y, results, predicted = domestic_foreign(df, True)
    plt.title(country)
    X, y, results, predicted = domestic_wopenth(df, True)
    X, y, results, predicted = domestic_foreign_wopenth(df)
    plt.title(country)
    X, y, results, predicted = domestic_widestth(df, True)
    X, y, results, predicted = domestic_genre(df)

if __name__ == "__main__":
    #initial_dataframe()
    df = unpickle(PICKLEDIR + "df")

    X, y, results, predicted = domestic_wreldate(df, True)

    wreldate_hist(df)

    domestic_origin_wopenth(df)
    
    domestic_origin_wopenth_wreldate(df)

    X, y, results, predicted = domestic_wopenth(df, True)
    
    countries = df["OriginC"].values
    counts = Counter(countries)
    countries = counts.most_common(COUNTRYLIM)
    countries = [("MEXICO", 29)]
    '''[('INDIA', 267), ('FRANCE', 150), ('CHINA', 48), ('ITALY', 31), 
    ('SPAIN', 30), ('MEXICO', 29), ('GERMANY', 28), ('SOUTH KOREA', 26), 
    ('JAPAN', 19), ('HONG KONG', 19)]'''
    
    for c in countries:
        print c[0]
        dfC = df.loc[df["OriginC"] == c[0]]
        run_specifics(dfC, c[0])
        test_split(dfC, c[0], r_state=2)#1,8,10 2is good for india, france

