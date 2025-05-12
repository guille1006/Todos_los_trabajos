import requests
from bs4 import BeautifulSoup
url = "http://www.imdb.com/title/tt0499549/"


HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

response = requests.get(url, headers= HEADERS)
print(response.status_code)



if response.status_code == 200:
    # Parseamos el contenido HTML con BeautifulSoup
    soup = BeautifulSoup(response.content, "lxml")  # o 'html.parser' si no tienes lxml
    title = soup.title.string
    meta = soup.find("meta", {"name": "description"}).get("content")
    actors = soup.find_all("a", {"data-testid": "title-cast-item__actor"})
    actors_list = [actor.get_text() for actor in actors]
    boxoffice_budget = soup.find("li", {"data-testid": "title-boxoffice-budget"})
    budget = boxoffice_budget.find("span", {"class": "ipc-metadata-list-item__list-content-item"}).get_text()

    print(budget)

print(title)
