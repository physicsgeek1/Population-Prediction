# import statements
from typing import Any, Union

import requests
import datetime
import numpy as np
import pandas as pd
import os
import sys


def get_weather(locationid, datasetid, begin_date, end_date, mytoken, base_url):
    token = {'token': mytoken}

    # passing as string instead of dict because NOAA API does not like percent encoding
    params = 'datasetid=' + str(datasetid) + '&' + 'locationid=' + str(locationid) + '&' + 'startdate=' + str(
        begin_date) + '-01-01' + '&' + 'enddate=' + str(end_date+1) + '-01-01' + '&' + 'limit=25' + '&' + 'units=standard'

    r = requests.get(base_url, params=params, headers=token)
    print("Request status code: " + str(r.status_code))

    try:
        # results comes in json form. Convert to dataframe
        df = pd.DataFrame.from_dict(r.json()['results'])
        print("Successfully retrieved " + str(len(df['station'].unique())) + " stations")
        dates = pd.to_datetime(df['date'])
        print("Last date retrieved: " + str(dates.iloc[-1]))

        return df

    # Catch all exceptions for a bad request or missing data
    except:
        print("Error converting weather data to dataframe. Missing data?")


def get_station_info(locationid, datasetid, mytoken, base_url):
    token = {'token': mytoken}

    # passing as string instead of dict because NOAA API does not like percent encoding

    stations = 'locationid=' + str(locationid) + '&' + 'datasetid=' + str(
        datasetid) + '&' + 'units=standard' + '&' + 'limit=1000'
    r = requests.get(base_url, headers=token, params=stations)
    print("Request status code: " + str(r.status_code))

    try:
        # results comes in json form. Convert to dataframe
        df = pd.DataFrame.from_dict(r.json()['results'])
        print("Successfully retrieved " + str(len(df['id'].unique())) + " stations")

        if df.count().max() >= 1000:
            print('WARNING: Maximum data limit was reached (limit = 1000)')
            print('Consider breaking your request into smaller pieces')

        return df
    # Catch all exceptions for a bad request or missing data
    except:
        print("Error converting station data to dataframe. Missing data?")


if __name__ == '__main__':
    stations = pd.DataFrame()
    mytoken = 'iWcBIsMUzseLjhZBmEnQGzxKAhyHMjCh'

    FIPSlist = ['01','02','04','05','06','08','09','10','11','12','13','15','16',
               '17','18','19','20','21','22','23','24','25','26','27','28','29','30',
                '31','32','33','34','36','37','38','39','40','41','42','44','45','46',
                '47','48','49','50','51','53','54','55','56']


    # This is for gathering station information and saving the info as a csv. update FIPSlist with what states are needed.
    # for fips in FIPSlist:
    #
    #      FIPS = 'FIPS:'+fips
    #      print(fips)
    #
    #      Test = get_station_info(FIPS, 'GSOY', mytoken, 'https://www.ncdc.noaa.gov/cdo-web/api/v2/stations')
    #
    #      output_file_name = '../../FIPS/' + FIPS + '.csv'
    #      Test.to_csv(output_file_name, index=False)


    for fips in FIPSlist:
        FIPS = 'FIPS:'+fips
        print(fips)

        imported = pd.read_csv('../../FIPS/' + FIPS + '.csv')

        if stations is None:
            stations = imported
        else:
            stations = stations.append(imported, ignore_index=True, sort=True)
        print(stations.head())
        print(stations.shape)
        output_file_name = '../../FIPS/StationListUS.csv'
        stations.to_csv(output_file_name, index=False)



