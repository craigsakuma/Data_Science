#!/usr/bin/python
"""
MOVIE SCRAPER
=============

This script is used to scrape movie titles from imdb using 
the kimono labs API
"""

import json
import urllib
import pandas as pd


def getMovies(year, api_key=api_key):
    """
    Creates list of top 50 movies by gross box office
    sales for a year with ratings and sales
    """
    
    movies, ratings, sales = [], [], []
    url = "https://www.kimonolabs.com/api/eeikle7m?" + \
            "apikey={}".format(api_key) + \
            "&release_date={year}".format(year=year) 
    
    data = json.load(urllib.urlopen(url))
    
    # Iterate through json object to collect data
    for n in xrange(data['count']):
        n_title = data['results']['collection1'][n]['title']['text']
        n_rating = data['results']['collection1'][n]['rating']
        n_sales = data['results']['collection1'][n]['sales']
        movies.append(n_title)
        ratings.append(n_rating)
        sales.append(n_sales)
    
    data = pd.DataFrame({'movie':movies,'rating':ratings,'sales':sales})
    
    return data


def getMoviesYears(year_list):
    "Collects movie data from list of years"
    # create empty DataFrame
    movies = pd.DataFrame(columns=['movie','rating','sales'])
    for year in year_list:
        temp_movies = getMovies(year)
        movies = pd.concat(movies,temp_movies)
    
    return movies


if __name__ == "__main__":
    api_key = 'enter your api key'
    example = getMovies('1999')
    print example
