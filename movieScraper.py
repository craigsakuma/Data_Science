
"""
Creates aliases for spans and zones

Requires installation of unidecode package
"""

import json
import urllib
import unicodedata
import sys
import re
import numpy as np
import pandas as pd
import datetime as dt
import redshift.tableLib as tl
import redshift.tablePermission as tablePermission
from unidecode import unidecode
import preprocessing.sqlTransactions as sqlT
from redshift import s3trx 
import redshift.bulkRedshift as blkr


def cleanText(text):
    "Removes punctuation and replaces spaces with underscores"
    text = unidecode(text)
    text = re.sub("[^A-Za-z0-9 _]","",text)
    text = text.replace(" ","_").lower()
    return text

    
def getMovies(year):
    "Creates list of movies by year (selects top 250 by box office gross)"
    movies = set()    
    for n in [1,51]:
        url = "https://www.kimonolabs.com/api/5zx3f8d8?" + \
                "apikey=TTBcrF02M3sETvHdHqP5Wxd3gpcPPfEC" + \
                "&release_date={}".format(year) + \
                "&sort=boxoffice_gross_us,desc" +\
                "&start={}&title_type=feature".format(n)
        
        data = json.load(urllib.urlopen(url))
        for n in xrange(data['count']):
            # link = data['results']['collection1'][n]['property1']['href']
            # m_code = re.search("(tt\d+)",link)
            m_title = data['results']['collection1'][n]['movie_title']['text']
            m_title = cleanText(m_title)
            if len(m_title) < 6 or len(m_title) >25:
                pass
            else:
                movies.add(m_title)
    return list(movies)

def getMoviesYears(year_list):
    movies = []
    for year in year_list:
        temp_movies = getMovies(year)
        movies = movies + temp_movies
    return list(set(movies))




