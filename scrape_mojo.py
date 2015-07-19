import requests
import time
import re
import pickle
import unicodedata
import json
from bs4 import BeautifulSoup
from datetime import datetime

######
# Notes:
# get movie names and links from first 15 pages 
# of boxofficemojo Foreign Language section
#
# Starting URLS
#"http://www.boxofficemojo.com/genres/chart/?id=foreign.htm"
#"http://www.boxofficemojo.com/genres/chart/?view=main&sort=
#        gross&order=DESC&pagenum=15&id=foreign.htm"
#
# (Title, Link) : [OriginC, Budget, DomLifeGross, ForLifeGross, LimRelDate, 
#                  LtdOpenTh, WrelDate, WOpenTh, WReleaseTh, Genre1,
#                  Genre2, Genre3, Genre4, Awards]
#
######

def connect(url):
    response = requests.get(url)
    code = response.status_code
    page = response.text
    return response, code, page

def connection_process(url):
    response, code, page = connect(url)
    while code != 200:
        response, code, page =  connect(url)
        time.sleep(1)
    return page

def pickle_stuff(filename, data):
    with open(filename, 'w') as picklefile:
        pickle.dump(data, picklefile)

def unpickle(filename):
    with open(filename, 'r') as picklefile:
        old_data = pickle.load(picklefile)
    return old_data

def deunicode(words):
    ascii_word = unicodedata.normalize('NFKC', words).encode('ascii','ignore')
    return ascii_word

def get_movie_pages(fullkeys):
    for key in fullkeys:
        title = key[0].replace(" ", "_")
        title = title.replace("/", "-")
        link = key[1]
        page = connection_process(BASEURL + link)
        pickle_stuff(HTMLDIR + title + "_mojo.pkl", page)
        time.sleep(0.1)

def get_movie_value(soup, field_name):
    '''Grab a value from boxofficemojo HTML. Takes a string attribute of a 
    movie on the page and returns the string in the next sibling object 
    (the value for that attribute) or None if nothing is found.
    '''
    obj = soup.find(text=re.compile(field_name))
    if not obj: 
        return None
    # this works for most of the values
    next_sibling = obj.findNextSibling()
    if next_sibling:
        value = next_sibling.text
        value = deunicode(value)
        return value
    else:
        return None

def parse_foreignlanguage_table(soup):
    '''retrieve information from the bomojo foreign language table'''
    tables = soup.find_all('table')
    linklist = []
    loclist = []
    for table in tables:
        rows = table.findAll('tr')
        for tr in rows:
            cols = tr.findAll('td')
            for col in cols:
                links = col.findAll('a', href=re.compile("movies"))
                if len(links) > 0:
                    for link in links:
                        location = link.findNextSibling(text = 
                                                        re.compile("\((.*?)\)"))
                        loclist.append(location)
                    linklist.append(links)
        if len(linklist) > 0: #only need the one table of links
            return linklist, loclist

def get_foreign_titles():
    '''get titles from the bomojo foreign language pages'''
    fullkeys = []
    fulllocs = {}
    for i in range(PAGENUM):
        page_n = i+1
        url = "http://www.boxofficemojo.com/genres/chart/?view=main&sort=
        gross&order=DESC&pagenum=%d&id=foreign.htm" %page_n
        
        page = connection_process(url)
        soup = BeautifulSoup(page)
        
        linklist, loclist = parse_foreignlanguage_table(soup)
        links = linklist[0]
        keys = []
        movies = {}
        for l in range(len(links)):
            address = links[l]['href']
            title = deunicode(links[l].text)
            key = (title, address)
            fullkeys.append(key)
            keys.append(key)
            location = loclist[l]
            if location is not None:
                notuni = deunicode(loclist[l])
                location = notuni.replace("(","").replace(")","")
            movies[key] = location
            fulllocs[key] = location        
        pickle_stuff(PICKLEDIR + "keys_%d.pkl" %page_n, keys)
        pickle_stuff(PICKLEDIR + "locdict_%d.pkl" %page_n, movies)
    pickle_stuff(PICKLEDIR + "fullkeys.pkl", fullkeys)
    pickle_stuff(PICKLEDIR + "fulllocs.pkl", fulllocs)
    return fullkeys

def get_domestic_summary(soup):
    '''gets a list of data from the domestic summary box, since that table is 
    difficult to parse normally. concept from: https://github.com/skozilla/
    BoxOfficeMojo/blob/master/boxofficemojoAPI/movie.py '''

    center = soup.findAll("center")
    if len(center) == 0 or len(center) is None:
        pass
    table = center[0].find("table")
    tables = soup.find_all("div", "mp_box")
    data = []
    for table in tables:
        box_table_name = table.find("div", "mp_box_tab").string
        if box_table_name == "Domestic Summary":
            rows = table.findAll('tr')
            for tr in rows:
                cols = tr.findAll('td')
                if len(cols) >= 1:
                    for col in cols:
                        data.append(col.text)
    return data

def get_theater(theatersli):
    ''' gets number of theaters a movie was released in given a list from the 
    domestic summary box'''
    for t in range(len(theatersli)):
        if theatersli[t] == "theaters":
            theater = theatersli[t-1]
            if "(" in theater:
                theater = theater.replace("(","")
            return theater

def parse_messy_data(messy_data, soup):
    ''' parses the list of data received from the domestic summary box '''
    limdate = None
    widedate = None
    limtheaters = None
    widetheaters = None
    alltheaters = None
    for m in range(len(messy_data)):
        notuni = deunicode(messy_data[m]).strip()
        if notuni == "Release Dates:":
            valuemaybe = deunicode(messy_data[m+1])
            limdate = " ".join(valuemaybe.split()[:3])
            widedate = " ".join(valuemaybe.split()[4:7])
        elif notuni == "Opening Weekend:":
            widedate = get_movie_value(soup, "Release Date")
        if notuni == "Opening Weekend:" or notuni == "Wide Opening Weekend:":
            theatersli = deunicode(messy_data[m+2]).replace(",","").split()
            widetheaters = get_theater(theatersli)
        elif notuni == "Limited Opening Weekend:":
            theatersli = deunicode(messy_data[m+2]).replace(",","").split()
            limtheaters = get_theater(theatersli)
        elif notuni == "Widest Release:":
            theatersli = deunicode(messy_data[m+1]).replace(",","").split()
            alltheaters = get_theater(theatersli)
        if widedate is None:
            widedate = get_movie_value(soup, "Release Date")
    data = (limdate, limtheaters, widedate, widetheaters, alltheaters)
    return data

def get_nice_data(soup):
    budget = get_movie_value(soup, "Production Budget")
    main_genre = get_movie_value(soup, "Genre:")
    domestic = get_movie_value(soup, "Domestic Total")
    return budget, main_genre, domestic

def get_bomojo_values(fullkeys, fulllocs):
    '''combine all the bomojo data into a dictionary with movie title, link 
    keys'''
    movies= {}
    for key in fullkeys:
        title = key[0].replace(" ", "_")
        title = title.replace("/", "-")
        page = unpickle(HTMLDIR + title + "_mojo.pkl")
        soup = BeautifulSoup(page)
        tables = soup.find_all('table')
        budget, main_genre, domestic = get_nice_data(soup)
        if domestic is None:
            domestic = soup.find(text="Domestic:")
            if domestic is not None:
                domestic = deunicode(soup.find(text="Domestic:")
                                     .find_parent("td").find_next_sibling("td")
                                     .get_text(strip=True))
        foreign = soup.find(text="Foreign:")
        if foreign is not None:
            foreign = deunicode(soup.find(text="Foreign:").find_parent("td")
                                .find_next_sibling("td").get_text(strip=True))
        messy_data =  get_domestic_summary(soup)
        parsed_data = parse_messy_data(messy_data, soup)
        academy_str = soup.find(text=re.compile("Academy"))
        academy = False
        if academy_str is not None:
            academy = True
        genres = [main_genre]
        genre_result = soup.find_all(href=re.compile("genres/chart"))
        for genre in genre_result:
            genres.append(deunicode(genre.text))
            genres = set(genres)
        movies[key] = [fulllocs[key], budget, domestic, foreign, 
                       parsed_data[0], parsed_data[1], parsed_data[2], 
                       parsed_data[3], parsed_data[4], genres, academy]
    pickle_stuff(PICKLEDIR + "movies.pkl", movies)
    return movies

def get_omdb_countries(movies):
    '''fill in missing country data with values from OMDB'''
    for key in movies:
        vals = movies[key]
        if vals[0] is None:
            country = None
            title = key[0]
            title = "+".join(title.split())
            url = OMDBURL + title + OMDBOptions
            page = connection_process(url)
            json_p = json.loads(page)
            if "Country" in json_p:
                country =  json_p["Country"]
                if country is not None:
                    country_list = deunicode(country).split(", ")
                    print country_list
                    country = country_list[0]
                    if country == "USA" and len(country_list) > 1:
                        country = country_list[1]
                    print type(country), country
            vals[0] = country
            movies[key] = vals
    pickle_stuff(PICKLEDIR + "moviesfilled.pkl", movies)
    return movies

def clean_and_make_none(vals):
    '''remove difficult characters and change N/As to None'''
    newvals = []
    for v in range(len(vals)):
        item = vals[v]
        if item == "N/A" or item == "n/a" or item == "na":
            item = None
        if item is not None and type(item) == str:
            if "," in item:
                item = item.replace(",","")
            if "$" in item:
                item = item.replace("$","")
            if "million" in item:
                vallist = item.split()
                item = float(vallist[0])*(10**6)
        newvals.append(item)
    return newvals
    
def clean_data(movies):
    for key in movies:
        vals = movies[key]
        vals = clean_and_make_none(vals)
        movies[key] = vals
    pickle_stuff(PICKLEDIR + "cleanmovies.pkl", movies)
    return movies
        

PAGENUM = 15
PICKLEDIR = "./pkls/"
HTMLDIR = "./bomojo/"
BASEURL = "http://www.boxofficemojo.com/"
'''
#get_foreign_titles()
'''
fullkeys = unpickle(PICKLEDIR + "fullkeys.pkl")
fulllocs = unpickle(PICKLEDIR + "fulllocs.pkl")
#smallkeys = unpickle(PICKLEDIR + "keys_1.pkl")
'''
#get_movie_pages(fullkeys)
'''
#movies = get_bomojo_values(fullkeys, fulllocs)

#OMDBURL = "http://www.omdbapi.com/?t="
#OMDBOptions = "&y=&plot=short&r=json"
#example title: Instructions+Not+Included
#movies = unpickle(PICKLEDIR + "movies.pkl")
#movies = get_omdb_countries(movies)

movies = unpickle(PICKLEDIR + "moviesfilled.pkl")

print clean_data(movies)
