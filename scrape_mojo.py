import requests
import time
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


print soup.find('a')['href'] #href="/movies/')

print code
