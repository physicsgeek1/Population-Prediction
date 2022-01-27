import pandas as pd
import numpy as np
import requests
import datetime
import os
import sys


def get_weather(stationid, begin_date, end_date, mytoken, base_url):
    token = {'token': mytoken}

    # passing as string instead of dict because NOAA API does not like percent encoding
    # params = {'stationid': stationid, 'datasetid': datasetid, 'statdate': f'{begin_date}-01-01',
    #           'enddate': f'{end_date}-01-01', 'limit': 1000}
    params =  'stationid=' + str(stationid) + '&' + 'startdate=' + str(begin_date) + '-01-01' + '&' +\
             'enddate=' + str(end_date) + '-01-01' + '&' + 'limit=1000' + '&' + 'units=metric' # + '&' + 'datasetid=' + str(datasetid)
    # print(params)
    # 'datatypeid=' + str(datasetid) +'&' +
    r = requests.get(base_url, params=params, headers=token)
    print("Request status code: " + str(r.status_code))

    try:
        # results comes in json form. Convert to dataframe
        df = pd.DataFrame.from_dict(r.json()['results'])
        # print(r.json()['results'])
        print("Successfully retrieved " + str(len(df['station'].unique())) + " stations")
        dates = pd.to_datetime(df['date'])
        print("Last date retrieved: " + str(dates.iloc[-1]))

        return df

    # Catch all exceptions for a bad request or missing data
    except:
        print("Error converting weather data to dataframe. Missing data?")


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    # reading in the files and appending all data into one dataframe
    df = pd.read_csv('../../Raw/US_Mammalia/Clean_population_record_for_select_Mammalia.csv')
    df2 = pd.read_csv('../../Raw/US_Amphibia/Clean_population_record_for_select_Amphibia.csv')
    df3 = pd.read_csv('../../Raw/US_Reptilia/Clean_population_record_for_select_Reptilia.csv')

    df = df.append(df2)
    df = df.append(df3)

    # filtering out all datasets that have less than 5 data points
    df.drop(columns=['Sub species'], inplace=True)
    df_filter = df[df.count(axis=1) > 26]  # keeps all rows with at least 5 data points

    # convert the df to a long format where each row corresponds to 1 data point (year and population)
    df_wide = df_filter.drop(columns=['Class', 'Order', 'Family', 'Genus', 'Species',
                                      'Common Name', 'Region', 'Decimal Latitude',
                                      'Decimal Longitude', 'Are coordinates for specific location?',
                                      'system', 'biome', 'realm', 'Native', 'Alien', 'Invasive',
                                      'Units', 'Sampling method', 'Data transformed'])

    df_wide2 = df_filter[['id', 'Class', 'Order', 'Family', 'Genus', 'Species',
                          'Common Name', 'Region', 'Decimal Latitude',
                          'Decimal Longitude', 'Are coordinates for specific location?',
                          'system', 'biome', 'realm', 'Native', 'Alien', 'Invasive',
                          'Units', 'Sampling method', 'Data transformed']]

    col_list = ['id']
    for col in df_wide.columns:
        if col != 'id':
            col_name = 'pop' + col
            col_list.append(col_name)

    df_wide.columns = col_list
    # print('df_wide',df_wide.shape, '\n', df_wide.head())
    df_long = pd.wide_to_long(df_wide, stubnames='pop', i=['id'],
                              j='year').reset_index()
    # print('df_long',df_long.shape, '\n', df_long.head())
    # recombining the data together to create a clean df
    df_clean = pd.merge(df_wide2, df_long, on='id', how='inner')
    df_clean = df_clean[df_clean['pop'].notnull()]
    # print('df_clean', df_clean.shape, '\n', df_clean.head())
    # print(df_clean.head())
    # print(df_clean.shape)

    # uncomment if needed: here are a couple of quick ways to check number of data point and unique lat long combos.
    # df_latlong = df_clean[['year', 'Decimal Latitude', 'Decimal Longitude']]
    # print(df_latlong.shape)
    # df_latlong = df_latlong.drop_duplicates()
    # print(df_latlong.shape)
    # latlongcount = df_clean[['Decimal Latitude', 'Decimal Longitude']]
    # latlongcount = latlongcount.drop_duplicates()
    # print(latlongcount.shape)

    # pull all of the columns from latlong needed to match to a station_id.
    df_latlong = df_clean[['id', 'year', 'Decimal Latitude', 'Decimal Longitude']]
    # print(df_latlong.shape)

    # calculate the min and max of the years that the study has data
    result = df_latlong.groupby('id').agg({'year': ['min', 'max']})
    result.columns = result.columns.map('_'.join)
    result = result.reset_index()
    # print(result.head())

    #attaches min and max years to df_latlong
    df_latlong = pd.merge(df_latlong, result, on='id', how='left')
    # print(df_latlong.shape)
    # print(df_latlong.head())

    # creates a list with id# lat and long as well as min and max year to search and pull necessary weather data
    df_unilatlong = df_latlong.drop(columns=['year'])
    df_unilatlong = df_unilatlong.drop_duplicates()
    df_unilatlong = df_unilatlong.reset_index()
    print('df_unilatlong', df_unilatlong.shape, '\n', df_unilatlong.head())

    # create the station_id dataframe in a way that it can be searched and used.
    stations = pd.read_csv('../../FIPS/StationListUS.csv')
    stations['mindate'] = pd.to_datetime(stations['mindate'], format='%Y-%m-%d')
    stations['maxdate'] = pd.to_datetime(stations['maxdate'], format='%Y-%m-%d')
    stations['minyear'] = stations['mindate'].dt.year
    stations['maxyear'] = stations['maxdate'].dt.year
    stations.rename(columns={'id': 'station_id'}, inplace=True)
    # print('stations', stations.shape, '\n', stations.head())

    # initialize some arrays so the for loop is more manageable
    ids = df_unilatlong['id']
    lats = df_unilatlong['Decimal Latitude']
    longs = df_unilatlong['Decimal Longitude']
    mins = df_unilatlong['year_min']
    maxs = df_unilatlong['year_max']
    station_id = pd.DataFrame(columns=['station_id'])

    for i in range(0, len(lats)):
        idloop = ids[i]
        latpop = lats[i]
        longpop = longs[i]
        Min = mins[i]
        Max = maxs[i]
        stationdatesort = stations.loc[stations['minyear'] <= Min]
        stationdatesort = stationdatesort.loc[stationdatesort['maxyear'] >= Max]
        stationdatesort = stationdatesort.loc[stationdatesort['datacoverage'] >= 0.7]
        stationdatesort = stationdatesort[abs(stationdatesort['latitude']).between(abs(latpop) - 5, abs(latpop) + 5)]
        stationdatesort = stationdatesort[abs(stationdatesort['longitude']).between(abs(longpop) - 5, abs(longpop) + 5)]
        if len(stationdatesort) == 0:
            # print('no data for '+ str(i) + ' lat= ' + str(latpop) + ' long= ' + str(longpop) + ' id= ' + str(idloop))
            station_id.loc[i, 'station_id'] = ''  # np.nan
            continue

        stationdatesort['Distance'] = np.sqrt((stationdatesort['latitude'] - latpop) ** 2 +
                                              (stationdatesort['longitude'] - longpop) ** 2)
        j = stationdatesort['Distance'].idxmin()
        station_id.loc[i, 'station_id'] = stationdatesort.at[j, 'station_id']

    df_unilatlong['Station_ID'] = station_id['station_id']
    print('df_unilatlong', df_unilatlong.shape, '\n', df_unilatlong.head())

    ################# The Code to pull all of the weather data from the API is in  the condenced comment here.
    # weather_data = pd.DataFrame()
    # for k in range(0, len(df_unilatlong)):
    #     stationid = df_unilatlong.at[k, 'Station_ID']
    #     datasetid = 'GSOY'
    #     begindate = df_unilatlong.at[k, 'year_min']
    #     enddate = df_unilatlong.at[k, 'year_max']
    #     mytoken = 'iWcBIsMUzseLjhZBmEnQGzxKAhyHMjCh'
    #     if stationid == '':
    #         continue
    #     base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GSOY'
    #     for l in range(begindate, enddate):
    #         begin_date = l
    #         end_date = l + 1
    #         temp = get_weather(stationid, begin_date, end_date, mytoken, base_url)
    #         weather_data = weather_data.append(temp, ignore_index=True, sort=True)
    #
    #         print(k, stationid, 'weather_data shape', weather_data.shape)
    #
    #     # print('temp', temp.head())
    # print(weather_data.head())
    # print('weather_data shape', weather_data.shape)
    # output_file_name = '../../Weather_api/WeatherData.csv'
    # weather_data.to_csv(output_file_name, index=False)

    #### Read in CSV of weather data and convert from long format to wide format for all of the measurables/attributes

    Weatherdata = pd.read_csv('../../Weather_api/WeatherData.csv')
    print('Weatherdata', Weatherdata.shape)
    Weatherdata2 = pd.read_csv('../../Weather_api/WeatherData2.csv')
    print('Weatherdata2', Weatherdata2.shape)
    Weatherdata = Weatherdata.append(Weatherdata2)
    print('Weatherdata', Weatherdata.shape)
    Weatherdata = Weatherdata.drop(columns=['attributes'])
    Weatherdata = Weatherdata.drop_duplicates()
    Wdata_wide = Weatherdata.pivot_table(index = ['station','date'], columns = 'datatype', values = 'value').reset_index()
    Wdata_wide['date'] = pd.to_datetime(Wdata_wide['date'])
    Wdata_wide['year'] = Wdata_wide['date'].dt.year
    Wdata_wide = Wdata_wide.drop(columns=['date'])
    print('Weatherdata', Weatherdata.shape, '\n', Weatherdata.head())
    print('Wdata_wide', Wdata_wide.shape, '\n', Wdata_wide.head())

    ### connect the Station ID data to the sutdy id so that it can be merged with pop data in df_clean
    joined = pd.merge(Wdata_wide, df_unilatlong[['id', 'Station_ID']], left_on='station', right_on='Station_ID')
    joined = joined.drop(columns=['station'])
    print('joined', joined.shape, '\n', joined.head())
    Final = pd.merge(df_clean,joined, on=['id', 'year'], how='left')
    print('Final', Final.shape, '\n', Final.head())
    output_file_name = '../../Weather_api/Final2.csv'
    Final.to_csv(output_file_name, index=False)

    Missing = Final[['id', 'Decimal Latitude', 'Decimal Longitude', 'year', 'Station_ID']]
    Missing = Missing[Missing['Station_ID'].isnull()]
    print('Missing', Missing.shape, '\n', Missing.head())

    ### Find the population values that do not have any associated weather data. Some will have to be dropped if a station is not close enough.
    # Missing = Final[['id', 'Decimal Latitude', 'Decimal Longitude', 'year', 'Station_ID']]
    # Missing = Missing[Missing['Station_ID'].isnull()]
    # Missing = Missing.reset_index()
    # print('Missing', Missing.shape, '\n', Missing.head())
    #
    # M_ids = Missing['id']
    # M_lats = Missing['Decimal Latitude']
    # M_longs = Missing['Decimal Longitude']
    # M_years = Missing['year']
    # station_id2 = pd.DataFrame(columns=['station_id'])
    #
    # #gather and attach all of the station id info to the missing data
    # for i in range(0, len(M_lats)):
    #     idloop2 = M_ids[i]
    #     latpop2 = M_lats[i]
    #     longpop2 = M_longs[i]
    #     year = M_years[i]
    #     stationds = stations.loc[stations['minyear'] <= year]
    #     stationds = stationds.loc[stationds['maxyear'] >= year]
    #     stationds = stationds.loc[stationds['datacoverage'] >= 0.7]
    #     stationds = stationds[abs(stationds['latitude']).between(abs(latpop2) - 5, abs(latpop2) + 5)]
    #     stationds = stationds[abs(stationds['longitude']).between(abs(longpop2) - 5, abs(longpop2) + 5)]
    #     if len(stationds) == 0:
    #         print('no data for '+ str(i) + ' lat= ' + str(latpop2) + ' long= ' + str(longpop2) + ' id= ' + str(idloop2))
    #         station_id.loc[i, 'station_id'] = ''  # np.nan
    #         continue
    #
    #     stationds['Distance'] = np.sqrt((stationds['latitude'] - latpop) ** 2 +
    #                                           (stationds['longitude'] - longpop) ** 2)
    #     j = stationds['Distance'].idxmin()
    #     station_id2.loc[i, 'station_id'] = stationds.at[j, 'station_id']
    #
    # Missing['Station_ID'] = station_id2['station_id']
    # print('Missing', Missing.shape, '\n', Missing.head())
    # Missing = Missing[Missing['Station_ID'].notnull()].reset_index()
    # print('Missing', Missing.shape, '\n', Missing.head())
    #
    # weather_data = pd.DataFrame()
    # for k in range(0, len(Missing)):
    #     stationid = Missing.at[k, 'Station_ID']
    #     datasetid = 'GSOY'
    #     begindate = int(Missing.at[k, 'year'])
    #     enddate = begindate + 1
    #     mytoken = 'iWcBIsMUzseLjhZBmEnQGzxKAhyHMjCh'
    #     base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GSOY'
    #     # for l in range(begindate, enddate):
    #     #     begin_date = l
    #     #     end_date = l + 1
    #     temp = get_weather(stationid, begindate, enddate, mytoken, base_url)
    #     weather_data = weather_data.append(temp, ignore_index=True, sort=True)
    #
    #     print(k, stationid, 'weather_data shape', weather_data.shape)
    #
    #     # print('temp', temp.head())
    # print(weather_data.head())
    # print('weather_data shape', weather_data.shape)
    # output_file_name = '../../Weather_api/WeatherData2.csv'
    # weather_data.to_csv(output_file_name, index=False)

