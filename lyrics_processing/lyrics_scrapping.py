from bs4 import BeautifulSoup
import pandas as pd

from urllib.request import Request, urlopen
import numpy
import csv

 
if __name__ == "__main__":
    
    # List the URLs here
    urllist = ["http://www.songlyrics.com/news/top-genres/hip-hop-rap/"]
    genre = urllist[0].split('/')[-1]
    req = Request(urllist[0], headers={'User-Agent': 'Mozilla/5.0'})
    doc = urlopen(req).read()
    soup = BeautifulSoup(doc, 'html.parser')
    div = soup.find( 'div', { 'class': 'box listbox' } )
    songs = div.find_all('a')
    songlinks = []
    for j in range(0,200):
        songlink = songs[j].get('href').encode('ascii','ignore')
        songlinks.append(songlink)

    songlinks = filter(None, songlinks)
    songlinks = [songlink for songlink in songlinks if (len(songlink.decode("utf-8").split('/'))==6)]

    lyricsvector = [] #input (bag of words)
    genrevector = [] #target
    songinfovector = []  #metadata (artist and songname)
    
    print("Scrapping Lyrics")
    for num, k in enumerate(range(0,len(songlinks))):
        
        if num % 1 == 0 : print("Processing Song",num)
        req = Request(songlinks[k].decode("utf-8"), headers={'User-Agent': 'Mozilla/5.0'})
        songdoc = urlopen(req).read()
        songsoup = BeautifulSoup(songdoc, 'html.parser')
        songinfo = songsoup.title.get_text().encode('ascii', 'ignore')
        songdiv = songsoup.find( 'div', { 'id': 'songLyricsDiv-outer' } )
        lyrics = songdiv.getText().replace("\'", "").replace("\r", " ")
        lyricsvector.append(lyrics)
        songinfovector.append(songinfo)


    df = pd.DataFrame([])

    for song_name ,lyrics in zip(songinfovector,lyricsvector):

        lyric_split = set(l for l in lyrics.split("\n") if len(l) != 0)
        df_temp = pd.DataFrame({"song":[song_name]*len(lyric_split), "lyrics": list(lyric_split)})
        df = df.append(df_temp)


    df_path = "../data/" + genre + ".csv"
    pickle_path = "../data/" + genre + ".pickle"


    # Save
    df.to_csv(df_path)

    with open(pickle_path,"wb") as f:
        pickle.dump(df.lyrics.values.tolist(),f)



    
