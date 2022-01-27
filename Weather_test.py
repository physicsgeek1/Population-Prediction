import pandas as pd
import numpy as np
import requests
import datetime
import os
import sys


def get_weather(locationid, datasetid, begin_date, end_date, mytoken, base_url):
    token = {'token': mytoken}

    # passing as string instead of dict because NOAA API does not like percent encoding
    params = 'stationid=' + str(locationid) + '&' + 'startdate=' + str(
        begin_date) + '&' + 'enddate=' + str(
        end_date) + '&' + 'limit=25'  # 'datasetid=' + str(datasetid) + '&' + '&' + 'units=standard'

    r = requests.get(base_url, params=params, headers=token)
    print("Request status code: " + str(r.status_code))

    try:
        # results comes in json form. Convert to dataframe
        df = pd.DataFrame.from_dict(r.json()['results'])
        test = pd.read_json
        print(r.json()['results'])
        print("Successfully retrieved " + str(len(df['station'].unique())) + " stations")
        dates = pd.to_datetime(df['date'])
        print("Last date retrieved: " + str(dates.iloc[-1]))

        return df
    # Catch all exceptions for a bad request or missing data
    except:
        print("Error converting weather data to dataframe. Missing data?")


def get_datatypes(datasetid, mytoken, base_url):
    token = {'token': mytoken}

    # passing as string instead of dict because NOAA API does not like percent encoding
    params = 'datasetid=' + str(datasetid)

    r = requests.get(base_url, params=params, headers=token)
    print("Request status code: " + str(r.status_code))

    try:
        # results comes in json form. Convert to dataframe
        df = pd.DataFrame.from_dict(r.json()['results'])
        test = pd.read_json
        print(r.json()['results'])
        print("Successfully retrieved " + str(len(df['station'].unique())) + " stations")
        dates = pd.to_datetime(df['date'])
        print("Last date retrieved: " + str(dates.iloc[-1]))

        return df
    # Catch all exceptions for a bad request or missing data
    except:
        print("Error converting weather data to dataframe. Missing data?")


if __name__ == '__main__':
    mytoken = 'iWcBIsMUzseLjhZBmEnQGzxKAhyHMjCh'

    # df = get_weather('GHCND:USW00027502', 'GSOY', '2008-01-01', '2009-01-01', mytoken,
    #                  'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GSOY')
    # print(df.head())

    df2 = get_datatypes('GSOY', mytoken, 'https://www.ncdc.noaa.gov/cdo-web/api/v2/datatypes')

    output_file_name = '../../Weather_api/DataTypes.csv'
    df2.to_csv(output_file_name, index=False)
