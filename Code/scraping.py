import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from lxml import etree


def scrape_trt_news():

    url = "https://www.trthaber.com"


    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    news_cards = soup.find_all('div', class_='standard-card')


    df = pd.DataFrame(columns=['link', 'title', 'text'])

    for card in news_cards:
        link_tag = card.find('a', class_='site-url')
        if link_tag:
            link = link_tag['href']
            title = link_tag['title']

            article_response = requests.get(link)
            article_soup = BeautifulSoup(article_response.content, 'html.parser')

            news_content = article_soup.find('div', class_='news-content')
            if news_content:
                paragraphs = news_content.find_all('p')
                text = ' '.join(paragraph.text for paragraph in paragraphs)
            else:
                text = 'Content not found'
            df.loc[len(df)] = [title, link, text]


    df = df[df['text'] != 'Content not found']
    df = df.reset_index(drop=True)

    return df


def remove_newlines(text):
    clean_text = text.replace('\n', ' ').replace('\r', '').strip()
    return clean_text


def scrape_cnn_articles(max_articles=10):
    main_url = "https://edition.cnn.com"
    response = requests.get(main_url)

    soup = BeautifulSoup(response.content, 'html.parser')

    dom = etree.HTML(str(soup))

    matches = dom.xpath("//div[@data-open-link]")

    links = []
    for match in matches:
        link = match.get('data-open-link')
        if 'videos' and 'live-news' not in link:
            links.append(main_url + link)

    links = links[:max_articles]
    df = pd.DataFrame(columns=['headline', 'link', 'text'])

    for link in links:
        response = requests.get(link)
        page_soup = BeautifulSoup(response.content, 'html.parser')
        dom = etree.HTML(str(page_soup))
        headline = dom.xpath('//h1[@data-editable="headlineText"]')
        headline = remove_newlines(headline[0].text)

        paragraphs = dom.xpath('//p[@data-component-name="paragraph"]')
        text = []
        for i in range(len(paragraphs)):
            text.append(paragraphs[i].text)

        text = ' '.join(text)
        text = remove_newlines(text)

        df.loc[len(df.index)] = [headline, link, text]
        time.sleep(2)

    return df
