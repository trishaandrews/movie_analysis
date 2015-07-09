import requests
import time
import re
from bs4 import BeautifulSoup

url = "http://www.boxofficemojo.com/genres/chart/?id=foreign.htm"

def connect(url):
    response = requests.get(url)
    code = response.status_code
    page = response.text
    return response, code, page

response, code, page = connect(url)

while code != 200:
   response, code, page =  connect(url)
   time.sleep(1)

soup = BeautifulSoup(page)


tables = soup.find_all('table')#href="/movies/')
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
        break

links = linklist[0]
#print tables[3].prettify()
#for index, item in enumerate(table):
#    if item.fetchNextSiblings('a') == []:
#        #print index, item
#        #item.decompose()


#for index, item in enumerate(table):
#    if item is not None:
#        print index, item, type(item)
#        linklist.append(item.find_all('a'))

#print soup.find_all('a', href=re.compile("movies"))


#something = soup.h1.find_next_siblings('table').a

#find_all('a', href=re.compile("movies"))

#print something
#for t in table:
#    print t
#    tsoup = BeautifulSoup(t)
#    #linklist.append(soup.find_all('a', href=re.compile("movies")))

print links
#print code
