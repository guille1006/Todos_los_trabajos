import csv
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import pandas as pd
import mrjob

MOVIES_DATA = "data_in/movie_data.csv"
main_fields=["movie_title", "title_year", "director_name", "actor_1_name", "language", "country", "color", "budget", "imdb_score", "movie_imdb_link"]
# Esta celda debe ser completada por el estudiante


def load_dataframe(csv_name):
    df = pd.read_csv(csv_name)
    return df

def fields_selected_dataframe(full_data_frame):
    return full_data_frame[main_fields]

tabla_completa = load_dataframe(MOVIES_DATA)
tabla_breve = fields_selected_dataframe(tabla_completa)

print(tabla_breve.groupby("director_name"))