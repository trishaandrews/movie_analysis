import requests
import time
import re
import pickle
from bs4 import BeautifulSoup

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
# (Title, Link) : (OriginC, DomLifeGross, ForLifeGross, RelDate, LtdOpen$, 
#                  LtdOpenTh, WOpen$, WOpenTh, WReleaseTh, (Genre), Budget, 
#                  Awards?)
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

def get_movie_pages(fullkeys):
    for key in fullkeys:
        title = key[0].replace(" ", "_")
        title = title.replace("/", "-")
        link = key[1]
        page = connection_process(BASEURL + link)
        pickle_stuff(HTMLDIR + title + "_mojo.pkl", page)
        time.sleep(0.1)

def get_movie_value(soup, field_name):
    '''Grab a value from boxofficemojo HTML
    
    Takes a string attribute of a movie on the page and
    returns the string in the next sibling object
    (the value for that attribute)
    or None if nothing is found.
    '''
    obj = soup.find(text=re.compile(field_name))
    if not obj: 
        return None
    # this works for most of the values
    next_sibling = obj.findNextSibling()
    if next_sibling:
        return next_sibling.text 
    else:
        return None

def parse_foreignlanguage_table(soup):
    tables = soup.find_all('table')

    linklist = []
    for table in tables:
        rows = table.findAll('tr')
        for tr in rows:
            cols = tr.findAll('td')
            for col in cols:
                links = col.findAll('a', href=re.compile("movies"))
                if len(links) > 0:
                    linklist.append(links)
        if len(linklist) > 0:
            return linklist

def get_foreign_titles():
    
    for i in range(PAGENUM):
        page_n = i+1
        url = "http://www.boxofficemojo.com/genres/chart/?view=main&sort=gross&order=DESC&pagenum=%d&id=foreign.htm" %page_n
        
        page = connection_process(url)
        soup = BeautifulSoup(page)
        
        linklist = parse_foreignlanguage_table(soup)
        links = linklist[0]
        keys = []
        for link in links:
            address = link['href']
            title = link.text
            keys.append((title, address))
            fullkeys.append((title, address))
            
        pickle_stuff(PICKLEDIR + "keys_%d.pkl" %page_n, keys)
    pickle_stuff(PICKLEDIR + "fullkeys.pkl", fullkeys)

    return fullkeys

def get_domestic_summary(soup):
    #https://github.com/skozilla/BoxOfficeMojo/blob/master/boxofficemojoAPI/movie.py

    center = soup.findAll("center")

    if len(center) == 0:
        pass

    table = center[0].find("table")

    if len(center) is None:
        pass
    '''
    rows = table.findAll('tr')
    for tr in rows:
        cols = tr.findAll('td')
        contents = [a.renderContents() for a in cols]
        for con in contents:
            keyval = con.split(":")
            key = keyval[0]
            val = keyval[1].strip()
     '''      
    tables = soup.find_all("div", "mp_box")
    data = []
    for table in tables:
        box_table_name = table.find("div", "mp_box_tab").string
        if box_table_name == "Domestic Summary":
            rows = table.findAll('tr')
            for tr in rows:
                cols = tr.findAll('td')
                if len(cols) > 1:
                    for col in cols:
                        data.append(col.text)
    return data

def get_nice_data(soup):
    budget = get_movie_value(soup, "Production Budget")
    main_genre = get_movie_value(soup, "Genre:")
    reldate = get_movie_value(soup, "Release Date")
    domestic = get_movie_value(soup, "Domestic Total")
    return budget, main_genre, domestic

PAGENUM = 15
PICKLEDIR = "./pkls/"
HTMLDIR = "./bomojo/"
BASEURL = "http://www.boxofficemojo.com/"

fullkeys = []
#get_foreign_titles()
fullkeys = unpickle(PICKLEDIR + "fullkeys.pkl")

smallkeys = unpickle(PICKLEDIR + "keys_1.pkl")
#get_movie_pages(fullkeys)

for key in smallkeys:
    title = key[0].replace(" ", "_")
    title = title.replace("/", "-")
    page = unpickle(HTMLDIR + title + "_mojo.pkl")
    
    soup = BeautifulSoup(page)
    tables = soup.find_all('table')

    budget, main_genre, domestic = get_nice_data(soup)
    print budget
    print main_genre
    print domestic
    
    foreign = soup.find(text="Foreign:").find_parent("td").find_next_sibling("td").get_text(strip=True)
    print foreign
    
    messy_data =  get_domestic_summary(soup)
    print messy_data

    academy_str = soup.find(text=re.compile("Academy"))
    academy = False
    if academy_str is not None:
        academy = True
    print academy

    genres = []
    genre_result = soup.find_all(href=re.compile("genres/chart"))
    for genre in genre_result:
        genres.append(genre.text)

    print genres
    break
    
fulldata = {}

