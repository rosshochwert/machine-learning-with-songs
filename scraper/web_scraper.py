##########################################
# This script scrapes through Dave's music
# database and pulls the name and artist
# of each song listed. It then sends the
# name and artist of each song to LyricWiki
# where we then scrape the lyrics to each
# song. Threading used to speedup analysis.
#######################################


import requests
import ast
from bs4 import BeautifulSoup, Comment
from string import ascii_uppercase
from sets import Set
from multiprocessing import Pool
from multiprocessing import cpu_count
from itertools import chain
import urllib


def set_globvar_to_zero():
    global globvar
    globvar = 0


def increment_globvar():
    global globvar
    globvar = globvar + 1


def print_globvar():
    print globvar

#scrape


def scrape_music_db(url):
    data = requests.get(url).text  # pull data
    table_data = []  # create dictionary
    soup = BeautifulSoup(data, "lxml")  # create scraper

    song_table = soup.find("table", {"class": "music"})  # search for music table
    song_body = song_table.find('tbody')
    rows = song_body.find_all('tr')

    #genre_set = Set([]) #create a genre set

    #Iterate through all rows and columns in the table
    for row in rows[1:]:  # iterate through rows, skipping the first
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        td_artist = cols[1]
        td_title = cols[2]
        td_time = cols[3]
        td_bpm = cols[4]
        td_year = cols[5]
        td_genre = cols[6]
        table_data.append([td_artist, td_title, td_time, td_bpm, td_year, td_genre])  # get rid of empty values

    return table_data


#Analyze # of Songs in Each Genre
def genre_count(song_db):

    #Create Genre Set
    genre_set = Set([])
    for songs in song_db:
        genre_set.add(songs[5])
    genre_set = sorted(genre_set)
    #Analyze # of Songs in Each Genre
    for genre in genre_set:
        counter = 0
        for songs in song_db:
            if (songs[5] == genre):
                counter = counter + 1
        print "    %s : %i" % (genre, counter)


def get_lyric_url(artist, song):

    url = "http://lyrics.wikia.com/api.php?func=getSong"
    url += "&artist=" + urllib.quote(artist)
    url += "&song=" + urllib.quote(song)
    url += "&fmt=json"
    response = requests.get(url).text
    stripped = response.replace("song = ", "")
    json_song = ast.literal_eval(stripped)

    lyric_url = json_song["url"]
    return lyric_url


def scrape_lyricwiki(data):
    data_list = []
    if "amp;action=edit" in data[3]:
        print "    invalid"
        data_list = [data[0], data[1], data[2], ""]
    else:
        print "    valid"
        data = requests.get(data[3]).text
        soup = BeautifulSoup(data, "lxml")

        try:
            lyric_box = soup.find("div", {"class": "lyricbox"})
            for element in lyric_box(text=lambda text: isinstance(text, Comment)):
                element.extract()

            scripts = lyric_box.findAll("script")
            [s.extract() for s in scripts]
            bold = lyric_box.findAll("b")
            [s.extract() for s in bold]

            lyrics = lyric_box.renderContents().replace("<br/>", "\n")
            lyrics = lyrics.replace('<div class="lyricsbreak"></div>', '')
            data_list = [data[0], data[1], data[2], lyrics]
        except:
            print "FOUND EXCEPTION"
            print_globvar()
    return data_list


def unique_items(L):
    found = set()
    for item in L:
        if item[0] not in found:
            yield item
            found.add(item[0])

if __name__ == "__main__":

    set_globvar_to_zero()
    #Pool initialization
    pool = Pool(cpu_count() * 2)  # Creates a pool integrating cpu count

    #Creation of URL links
    urls = []  # Create array of urls
    url_0 = "http://www.cs.ubc.ca/~Davet/music/title/0.html"  # include 0-url
    urls.append(url_0)
    for c in ascii_uppercase:
        url = "http://www.cs.ubc.ca/~Davet/music/title/" + c + ".html"
        urls.append(url)

    #Begin web scraping
    print "Scraping Dave's music database...."
    results = pool.map(scrape_music_db, urls)

    pool.close()
    pool.join()

    song_database = list(chain.from_iterable(results))

    lyrics = []

    for song in song_database:
        lyric = get_lyric_url(song[0], song[1])
        print "got lyric"
        if lyric not in lyrics:
            lyrics.append(lyric)

    print "...Database completed"
    print "----"

    print "Removing duplicates..."
    print "length: %i" % (len(song_database))
    song_database = list(unique_items(song_database))
    print "length: %i" % (len(song_database))

    print "...duplicates removed."

    #Genre Analysis
    print "Analyzing genres..."
    genre_count(song_database)
    print "...Genre analysis completed"

    #LyricGathering
    print "Gathering lyrics..."
    lyricwiki_list = []
    for i in range(100):
        lyricwiki_list.append([song_database[i][0], song_database[i][1], song_database[i][5], get_lyric_url(song_database[i][0], song_database[i][1])])
        if("<" in lyricwiki_list[i][0] or "!" in lyricwiki_list[i][0]):
            print "found!!"
        if("<" in lyricwiki_list[i][1] or "!" in lyricwiki_list[i][1]):
            print "found!!"
    pool_lyricwiki = Pool(cpu_count() * 2)

    results_lyricwiki = pool_lyricwiki.map(scrape_lyricwiki, lyricwiki_list)

    pool_lyricwiki.close()
    pool_lyricwiki.join()

    lyric_database = list(chain.from_iterable(results_lyricwiki))

    print "...Gathering lyrics completed"
