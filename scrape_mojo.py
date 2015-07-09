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
# URLS
#"http://www.boxofficemojo.com/genres/chart/?id=foreign.htm"
#"http://www.boxofficemojo.com/genres/chart/?view=main&sort=
#        gross&order=DESC&pagenum=15&id=foreign.htm"
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
    with open(filename, 'w') as pickelfile:
        pickle.dump(data, pickelfile)

def unpickel(filename):
    with open(filename, 'r') as pickelfile:
        old_data = pickel.load(pickelfile)
    return old_data

def parse_foreignlanguage_table(tables):
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
    PAGENUM = 15
    fullkeys = []
    PICKLEDIR = "./pkls/"
    
    for i in range(PAGENUM):
        page_n = i+1
        url = "http://www.boxofficemojo.com/genres/chart/?view=main&sort=gross&order=DESC&pagenum=%d&id=foreign.htm" %page_n
        
        page = connection_process(url)
        soup = BeautifulSoup(page)
        tables = soup.find_all('table')
        
        linklist = parse_foreignlanguage_table(tables)
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

print get_foreign_titles()

