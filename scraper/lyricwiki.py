##############################################
# Test script to scrape lyrics from lyricwiki.
# It's not six-sigma, so a few glitches may occur.
##############################################


import requests, ast, re
from bs4 import BeautifulSoup, Comment


def get_lyric_url(artist, song):

    url = "http://lyrics.wikia.com/api.php?func=getSong"
    url += "&artist=" + artist
    url += "&song=" + song
    url += "&fmt=json"

    response = requests.get(url).text
    stripped = response.replace("song = ", "")
    json_song = ast.literal_eval(stripped)

    lyric_url = json_song["url"]
    return lyric_url


def scrape(url):
    data = requests.get(url).text
    soup = BeautifulSoup(data, "lxml")

    lyric_box = soup.find("div", {"class": "lyricbox"})
    for element in lyric_box(text=lambda text: isinstance(text, Comment)):
        element.extract()

    scripts = lyric_box.findAll("script")
    [s.extract() for s in scripts]
    bold = lyric_box.findAll("b")
    [s.extract() for s in bold]

    lyrics = lyric_box.renderContents().replace("<br/>","\n")
    lyrics = lyrics.replace('<div class="lyricsbreak"></div>','')

    return lyrics

if __name__ == "__main__":
    url = get_lyric_url("the lonely island", "jack sparrow")
    lyrics = scrape(url)
    print lyrics
