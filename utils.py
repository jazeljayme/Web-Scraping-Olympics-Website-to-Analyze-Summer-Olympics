import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json
import sqlite3
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from numpy.testing import assert_equal
from sqlalchemy import create_engine
import warnings
import os
warnings.filterwarnings('ignore')


dire = os.path.join('/home/msds2022/macot/dmw_lab/lab2/InitialCompilation/', 
                    'Olympics.db')
data_dir = '/home/msds2022/macot/dmw_lab/lab2/InitialCompilation/'

engine = create_engine("sqlite:///" + dire)


def get_stats_olympic():
    """Return the basic statistics of the `olympics` table."""

    with engine.connect() as conn:
        olympics = pd.read_sql("""SELECT * FROM olympics""",
                               conn)
        info = pd.read_sql("""SELECT * FROM info""",
                           conn)
        winners = pd.read_sql("""SELECT cast(rank as integer) rank,
                                        nation,
                                        cast(gold as integer) gold,
                                        cast(silver as integer) silver,
                                        cast(bronze as integer) bronze,
                                        cast(total as integer) total,
                                        year FROM winners""",
                              conn)
        sports = pd.read_sql("""SELECT * FROM sports""",
                             conn)
        participants = pd.read_sql("""SELECT * FROM PARTICIPANTS""",
                                   conn)
        gdp = pd.read_sql("""SELECT * FROM gdp""",
                          conn)
    data = pd.DataFrame(olympics.groupby('Country').count()
                        .sort_values('Year', ascending=False)['Year'][:4])

    display(olympics[['Athletes', 'Teams', 'Events']].describe())
    return olympics, info, winners, sports, participants, gdp


def get_continents_loc():
    """Return a DataFrame containing historical Olympics data."""

    with engine.connect() as conn:
        df_olympics = pd.read_sql("""
        SELECT *
        FROM OLYMPICS
        """, conn)

    Olympics_2020 = {'Olympic Games': 'Tokyo2020', 'Date':
                     'July 23 - August 8',
                     'Country': 'Japan', 'Athletes': 11090, 'Teams': 205,
                     'Events': 339, 'Year': 2021}
    df_olympics = df_olympics.append(Olympics_2020, ignore_index=True)
    continents = ['Europe',
                  'Europe',
                  'North America',
                  'Europe',
                  'Europe',
                  'Europe',
                  'Europe',
                  'Europe',
                  'North America',
                  'Europe',
                  'Europe',
                  'Europe',
                  'Australia',
                  'Europe',
                  'Asia',
                  'North America',
                  'Europe',
                  'North America',
                  'Europe',
                  'North America',
                  'Asia', 'Europe',
                  'North America',
                  'Australia',
                  'Europe',
                  'Asia',
                  'Europe',
                  'South America',
                  'Asia']
    df_olympics['Continents'] = continents
    df_olympics = df_olympics.replace(['Federal Republic of Germany'],
                                      'Germany')
    df_olympics = df_olympics.replace(['Australia, Sweden'], 'Australia')

    return df_olympics


def visualize_hosts(df_olympics):
    """Plot the number of times countries hosted the Olympics.

    Parameters:
    -----------
    df_olympics: pandas DataFrame
        The dataframe that contains the historical Olympics data.

    Returns:
    --------
    matplotlib figure
    """

    df_olympics['Country'].value_counts().plot(kind='barh', figsize=(9, 7),
                                               color='blue')
    plt.title('Figure 1. Number of times countries hosted the Olympics',
              fontsize=15)
    plt.xlabel('Counts', fontsize=14)
    plt.ylabel('Countries', fontsize=14)
    plt.show()


def visualize_continents(df_olympics, i=0, j=0, title=' '):
    """Plot the number of times continents hosted the Olympics.

     Parameters:
    -----------
    df_olympics: pandas DataFrame
        The dataframe that contains the historical Olympics data.

    i : int
        The start index that will be used to slice the dataframe.

    j : int
        The end index that will be used to slice the dataframe.

    title : str
        The title of the plot.

    Returns:
    --------
    matplotlib figure
    """

    df_olympics['Continents'][i:j].value_counts().plot(kind='barh',
                                                       figsize=(9, 5),
                                                       color='blue')
    plt.title(title, fontsize=15)
    plt.xlabel('Counts', fontsize=14)
    plt.ylabel('Continents', fontsize=14)
    plt.show()


def get_teams():
    """Return a Pandas DataFrame containing number of teams for every Summer
    Olympics.
    """

    with engine.connect() as conn:
        df = pd.read_sql('SELECT * FROM info', conn)

    # filter by number of events
    df = df.loc[df['label'] == 'Nations']
    df['value'].replace({r'\D.*': ''}, inplace=True, regex=True)
    df['value'] = pd.to_numeric(df['value'])
    df['year'] = pd.to_numeric(df['year'])
    return df.reset_index()


def visualize_teams(df_teams, i, j, title):
    """Plot the trend of the participating teams per Olympics event.

    Parameters:
    -----------
    df_teams : pandas DataFrame
        The data frame that contains the number of teams joined
        per year.

    i : int
        The start index that will be used to slice the dataframe.

    j : int
        The end index that will be used to slice the dataframe.

    title : str
        The title of the plot.

    Returns
    ---------
    matplotlib figure
    """

    plt.figure(figsize=(9, 5))
    plt.plot(df_teams['year'].iloc[i:j], df_teams['value'].iloc[i:j],
             'o-', c='blue')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Teams', fontsize=14)
    plt.title(title, fontsize=15)
    plt.show()


def get_sports():
    """Return dataframe containing the sports for every Summper Olympics."""

    with engine.connect() as conn:
        return pd.read_sql_query("""SELECT year, COUNT(events) AS events
                            FROM (SELECT DISTINCT year,
                                            TRIM(events) events
                            FROM sports) A GROUP BY year
                            """, conn).set_index('year')


def visualize_sports(year_trends):
    """Plot the trend of the number of sports for Summer Olympics.

    Parameters:
    -----------
    year_trends : pandas DataFrame
        The dataframe that contains the sports for every Summper
        Olympics.

    Returns:
    --------
    matplotlib figure
    """

    year_trends.plot(figsize=(14, 5), color='blue', marker='o')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Events', fontsize=14)
    plt.title('Figure 10. Olympics Sporting Events', fontsize=15)
    plt.show()


def create_sports_table():
    """Plot the number of unique sports/events per Summer Olympics."""

    conn = sqlite3.connect('Olympics.db')
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TEMPORARY TABLE tempsports
                AS SELECT year,events FROM sports;
            """)

    cursor.execute("UPDATE tempsports SET events =\
                          TRIM(REPLACE(events, CHAR(10), ' '))")
    cursor.execute("UPDATE tempsports SET events =\
                          'Basketball' WHERE events =\
                          'Basketball Basketball'")
    cursor.execute("UPDATE tempsports SET events =\
                          'Kata' WHERE events = 'Karate Kata'")
    cursor.execute("UPDATE tempsports SET events =\
                          'BMX Racing' WHERE events = 'Cycling BMX'")
    cursor.execute("UPDATE tempsports SET events =\
                          '3x3 Basketball' WHERE events = '3'")
    cursor.execute("UPDATE tempsports SET events =\
                          'Canoeing' WHERE events =\
                          'Canoeing Slalom'")
    cursor.execute("UPDATE tempsports SET events =\
                          'Cycling' WHERE events = 'Road cycling'")
    cursor.execute("UPDATE tempsports SET events =\
                          'Cycling' WHERE events = 'Track cycling'")

    df = pd.read_sql_query("""
                SELECT year, count(events) events FROM (
                SELECT DISTINCT s1.year, events
                FROM tempsports s1
                WHERE events NOT IN (SELECT events
                FROM tempsports s2 WHERE s2.year<s1.year)) a
                GROUP BY year
                """, conn).set_index('year')
    conn.close()

    df.plot(figsize=(14, 5), color='blue', marker='o')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Unique Events', fontsize=14)
    plt.title('Figure 11. Number of Unique Sporting Events', fontsize=15)
    plt.show()


def create_2020_events():
    """Return a DataFrame containing unique events in 2020 Olympics."""

    with engine.connect() as conn:
        df = pd.read_sql_query("""
            SELECT DISTINCT
            TRIM(REPLACE(events, CHAR(10), ' ')) events
            FROM sports WHERE year=2020
            AND TRIM(REPLACE(events, CHAR(10), ' '))
                    NOT IN (SELECT TRIM(REPLACE(events, CHAR(10), ' '))
                    FROM sports WHERE year<2020)
            AND events NOT LIKE '%cycling%'
            AND events NOT LIKE '%basketball%'
            AND LOWER(events) NOT LIKE '%canoeing%'
            """, conn)

    df.loc[df.events == '3', 'events'] = '3x3 Basketball'
    df.loc[df.events == 'Basketball\nBasketball', 'events'] = 'Basketball'
    df.loc[df.events == 'Cycling BMX', 'events'] = 'BMX Racing'
    df.loc[df.events == 'Karate Kata', 'events'] = "Kata"
    df = df.drop_duplicates()

    return df


def create_athletes():
    """Return a DataFrame containing the total athletes by gender per year."""

    with engine.connect() as conn:
        olympics = pd.read_sql("""SELECT * FROM olympics""",
                               conn)
        info = pd.read_sql("""SELECT * FROM info
        where label = 'Athletes'""",
                           conn)
    gender = dict()
    for i, values in (info[['value', 'year']]).iterrows():
        gender[values['year']] = [int(re.findall(r'([\d,]*\d+)',
                                                 values['value'])[0]
                                      .replace(',', '')),
                                  int(re.findall(r'([\d,]*\d+)',
                                                 values['value'])[0]
                                      .replace(',', '')) -
                                  int(re.findall(r'([\d,]*\d+)',
                                                 values['value'])[1]
                                      .replace(',', ''))]

    gender_df = pd.DataFrame.from_dict(gender, orient='index')
    gender_df.loc[1896, 1] = 0
    gender_df.loc[2016, [0, 1]] = [6179, 5059]
    gender_df.loc[2020, [0, 1]] = [5651, 5386]
    gender_df["total"] = gender_df[0].values + gender_df[1].values
    gender_df = gender_df.reset_index()
    gender_df.columns = ["year", "male", "female", "total"]

    return gender_df


def get_gender_trend(gender_df):
    """Plot the trend of the total athletes for Summer Olympics.

    Parameters:
    -----------
    gender_df : pandas DataFrame
        The dataframe containing the total athletes by gender per year.

    Returns:
    --------
    matplotlib figure
    """
    total_ath = pd.DataFrame(gender_df.set_index('year')
                             .pct_change()['total'])
    total_ath = total_ath[1:29]
    z = np.polyfit(gender_df.year, gender_df.total, 1)
    p = np.poly1d(z)

    plt.figure(figsize=(12, 5))
    plt.plot(gender_df.year, p(gender_df.year), "r--")
    plt.plot('year', 'total', data=gender_df, color='blue')
    plt.ylabel('Country', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.title('Figure 12. Year-on-Year number of athletes in Olympics',
              fontsize=15)
    plt.show()


def visualize_gender(gender_df):
    """Plot the gender split of participating atheletes per Olympics event.

    Parameters:
    -----------
    gender_df : pandas DataFrame
        The dataframe containing the total athletes by gender per year.

    Returns:
    --------
    matplotlib figure
    """
    plt.figure(figsize=(16, 5))
    plt.bar(gender_df.year, gender_df.male, label='Men',\
            color='blue', width=3)
    plt.bar(gender_df.year, gender_df.female, bottom=gender_df.male,
            label='Women', color='#FFD700', width=3)
    plt.title('Figure 13. Gender Split of the Athletes Over the Years',
              fontsize=15)
    plt.ylabel('Number of Athletes', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend()
    plt.show()


def get_winners():
    """Return a DataFrame for the top 10 countries for 2020 Olympics.

    Return a DataFrame containing the GDP, gini, happiness index, gold
    rate, conversion rate, and golden convert for the top 10 countries
    for 2020 Olympics.
    """

    with engine.connect() as conn:
        df_win = pd.read_sql("""
                    SELECT * FROM WINNERS;
                """, conn)
        df_gdp = pd.read_sql("""
                    SELECT * FROM GDP;
                """, conn)
        df_part = pd.read_sql("""
                    SELECT * FROM PARTICIPANTS;
                """, conn)

    df = df_win[df_win['year'] == 2020]
    df = df.reset_index(drop=True)

    df[['rank', 'gold', 'silver', 'bronze', 'total']] = df[[
        'rank', 'gold', 'silver', 'bronze', 'total']].astype(int)

    df_gdp['2020_un_gdp*'].replace(',', '', regex=True, inplace=True)
    df_gdp['2020_un_gdp*'] = df_gdp['2020_un_gdp*'].astype(int)

    df['nation'].replace('ROC', 'Russia', inplace=True)
    df['nation'].replace('Great Britain', 'United Kingdom', inplace=True)

    df_gdp['country'] = df_gdp['country'].str.strip()

    gdp_data = []
    countries = [
        'United States', 'China', 'Japan', 'United Kingdom', 'Russia',
        'Australia', 'Germany', 'Netherlands', 'Italy', 'France']
    for country in countries:
        gdp_data.append(df_gdp[df_gdp['country'] == country]
                        ['2020_un_gdp*'].values[0])
    df['gdp_nominal_2020'] = gdp_data

    # merge gini index by country
    df_gini = pd.read_csv(data_dir + 'worldbank_gini.csv')
    gini_data = []
    countries = [
        'United States',
        'China',
        'Japan',
        'United Kingdom',
        'Russian Federation',
        'Australia',
        'Germany',
        'Netherlands',
        'Italy',
        'France']
    for country in countries:
        gini_data.append(df_gini[df_gini['country_name'] == country]
                         ['mean_gini_index_wb'].values[0])
    df['gini_index'] = gini_data

    # merge happiness index by country
    df_hi = pd.read_csv(data_dir + 'happiness_index.csv')
    hi_data = []
    countries = [
        'United States', 'China', 'Japan', 'United Kingdom', 'Russia',
        'Australia', 'Germany', 'Netherlands', 'Italy', 'France']
    for country in countries:
        hi_data.append(df_hi[df_hi['country'] == country]['score'].values[0])
    df['happiness_index'] = hi_data

    # Get win rate
    part_data = []
    countries = [
        'United States', 'China', 'Japan', 'Great Britain', 'ROC',
        'Australia', 'Germany', 'Netherlands', 'Italy', 'France']
    for country in countries:
        part_data.append(df_part[df_part['Country'] == country]
                         ['Participants'].values[0])

    df['athlete_count'] = part_data

    df['gold_rate'] = df['gold'] / df['total']
    df['conversion_rate'] = df['total'] / df['athlete_count']
    df['golden_convert'] = df['gold'] / df['athlete_count']
    df.rename(columns={'total': 'total_medals'}, inplace=True)
    df.drop(['year', 'athlete_count'], axis=1, inplace=True)
    return df


def visualize_medals(df_winners):
    """Plot the number of medals won by the top 10 countries for 2020 Olympics

    Parameters:
    -----------
    df_winners : pandas DataFrame
        The dataframe containing data for the top 10 countries for 2020
        Olympics.

    Returns:
    --------
    matplotlib figure
    """

    plt.figure(figsize=(12, 5))
    ranked_total = df_winners.sort_values('total_medals', ascending=False)
    plt.barh(ranked_total['nation'], ranked_total['total_medals'],
             color='#FFD700')
    plt.ylabel('Countries', fontsize=14)
    plt.xlabel('Total of medals won', fontsize=14)
    plt.title('Figure 15. Total medals won by Countries', fontsize=15)
    plt.show()


def visualize_gold_counts(df_winners):
    """Plot the number of gold medals won by the top 10 countries for 2020
    Olympics.

    Parameters:
    -----------
    df_winners : pandas DataFrame
        The dataframe containing data for the top 10 countries for 2020
        Olympics.

    Returns:
    --------
    matplotlib figure
    """

    plt.figure(figsize=(12, 5))
    ranked_gold = df_winners.sort_values('gold', ascending=False)
    plt.barh(ranked_gold['nation'], ranked_gold['gold'],
             color='#008000')
    plt.ylabel('Countries', fontsize=14)
    plt.xlabel('Total of Gold medals won', fontsize=14)
    plt.title('Figure 16. Total Gold medals won by Countries', fontsize=15)
    plt.show()


def visualize_rank_conversion(df_winners):
    """Plot the medal conversion rate of the top 10 countries for 2020
    Olympics.

    Parameters:
    -----------
    df_winners : pandas DataFrame
        The dataframe containing data for the top 10 countries for 2020
        Olympics.

    Returns:
    --------
    matplotlib figure
    """

    plt.figure(figsize=(12, 5))
    ranked_conversion = df_winners.sort_values('conversion_rate',
                                               ascending=False)
    plt.barh(ranked_conversion['nation'],\
             ranked_conversion['conversion_rate'],
             color='#B22222')
    plt.ylabel('Countries', fontsize=14)
    plt.xlabel('Total of Gold medals won', fontsize=14)
    plt.title('Figure 17. Medal Conversion rate by Countries', fontsize=15)
    plt.show()


def visualize_golden_convert(df_winners):
    """Plot the golden conversion rate of the top 10 countries for 2020
    Olympics.

    Parameters:
    -----------
    df_winners : pandas DataFrame
        The dataframe containing data for the top 10 countries for 2020
        Olympics.

    Returns:
    --------
    matplotlib figure
    """

    plt.figure(figsize=(12, 5))
    ranked_conversion_gold = df_winners.sort_values('golden_convert',
                                                    ascending=False)
    plt.barh(ranked_conversion_gold['nation'],
             ranked_conversion_gold['golden_convert'], color='blue')
    plt.ylabel('Countries', fontsize=14)
    plt.xlabel('Golden Conversion Rate', fontsize=14)
    plt.title('Figure 18. Golden Conversion Rate by Countries', fontsize=15)
    plt.show()
