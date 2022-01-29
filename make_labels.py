# This script takes the consilidated data from Data_gathering.py and creates datasets
# based on percentage fill of measured weather parameters available. For all missing data,
# the missing attribute is approximated with the mean value of the available measured
# attributes over all years at that location. Then any training examples missing any 
# information are removed. Lastly, the datasets are exported as .csv to be used with 
# machine learning code.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.style.use('seaborn-white')
    pd.set_option('display.max_columns', None)

    # reads in .csv containing population and corresponding weather data
    df = pd.read_csv('Final2.csv')
    df = df[df['Station_ID'].notnull()].sort_values(by=['id','year'])
    
    # find unique study id's and initialize empty sets needed for scaling
    b = np.array([df['id']]).transpose()
    pop = np.array([df['pop']]).transpose()
    uid = np.unique(b)
    idpop = np.append(b,pop, axis=1)
    stdscale = []
    mmscale = []
    ss = StandardScaler()
    mm = MinMaxScaler()
    
    # the following for loop scales each study's measured populations as eithere
    # a standard scale (stdscale) or a minmix scalee (mmscale).
    for ids in uid:
        temp = idpop[idpop[:,0]==ids]
        temp2 = temp[:,1]
        #temp3 = temp[:,2]
        if len(stdscale) == 0:
            stdscale = ss.fit_transform(temp2.reshape(-1,1))
        else:
            ssscale = ss.fit_transform(temp2.reshape(-1,1))
            stdscale = np.append(stdscale, ssscale, axis=0)
            del ssscale

            
        if len(mmscale) == 0:
            mmscale = mm.fit_transform(temp2.reshape(-1,1))
        else:
            mscale = mm.fit_transform(temp2.reshape(-1,1))
            mmscale = np.append(mmscale, mscale, axis=0)
            del mscale  
    
    # Assign stdscale and mmscale to df.
    df['stdscale'] = stdscale
    df['mmscale'] = mmscale
    
    
    # df1 used to identify all features with <70% fill. Then any missing values 
    # are replaced with locational average of available data for the attribute.
    # Final result saved in df2.
    df1 = df.loc[:, df.isnull().mean() <= 0.3]
    meanlbl_df1 = ['TMAX', 'DP01', 'DP10', 'DP1X', 'DSNW', 'DT00', 'DT32', 'DX70', 
                  'DX90', 'EMNT', 'EMSN', 'EMXP', 'EMXT', 'PRCP', 'SNOW', 'TMIN']
    
    for i in meanlbl_df1:
        df1[i] = df1[i].fillna(df1.groupby('Station_ID')[i].transform('mean'))
   
    df2 = df1.dropna(axis = 0)
    
    
    # df3 used to identify all features with <80% fill. Then any missing values 
    # are replaced with locational average of available data for the attribute.
    # Final result saved in df4.    
    df3 = df.loc[:, df.isnull().mean() <= 0.2]  
    meanlbl_df2 = ['DP01', 'DP10', 'DP1X', 'EMXP', 'PRCP']
    
    for i in meanlbl_df2:
        df3[i] = df3[i].fillna(df1.groupby('Station_ID')[i].transform('mean'))

    df4 = df3.dropna(axis = 0)


    #PLOTS FOR data70
    plt.figure(1)
    plt.scatter(df2['year'], df2['pop'])
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.title('Populations - Dataset 2')
    plt.savefig('pop70.png')
    
    plt.figure(2)
    plt.scatter(df2['year'], df2['mmscale'])
    plt.xlabel('Year')
    plt.ylabel('Scaled Population')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.title('Scaled Population - Dataset 2')
    plt.savefig('popscaled70.png')


    #PLOTS FOR data80
    plt.figure(3)
    plt.scatter(df4['year'], df4['pop'])
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.title('Populations - Dataset 1')
    plt.savefig('pop80.png')
 
    plt.figure(4)
    plt.scatter(df4['year'], df4['mmscale'])
    plt.xlabel('Year')
    plt.ylabel('Scaled Population')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.title('Scaled Populations - Dataset 1') 
    plt.savefig('popscaled80.png')
    
    
    ######### groupby id, calculate percent change in pop, and remove any data that isnt an incremental year
    grouped = df2.groupby(['id']).apply(lambda x: x.sort_values(by=['year']))
    grouped['yeardiff'] = grouped['year'].diff()
    grouped['stddiff'] = grouped['stdscale'].diff()  # pct = percent, this gives results in decimal and not %
    grouped['mmdiff'] = grouped['mmscale'].diff()  # pct = percent, this gives results in decimal and not %
    grouped['lastpop'] = grouped['mmscale'].shift(1)
    df_out = grouped.reset_index(drop=True)
    df_out = df_out[df_out['yeardiff'] == 1]  # drops datapoints that are skip years

    output_file_name = 'data70.csv'
    #df_out.to_csv(output_file_name, index=False)

    grouped2 = df4.groupby(['id']).apply(lambda x: x.sort_values(by=['year']))
    grouped2['yeardiff'] = grouped2['year'].diff()
    grouped2['stddiff'] = grouped2['stdscale'].diff()  # pct = percent, this gives results in decimal and not %
    grouped2['mmdiff'] = grouped2['mmscale'].diff()  # pct = percent, this gives results in decimal and not %
    grouped2['lastpop'] = grouped2['mmscale'].shift(1)
    df_out2 = grouped2.reset_index(drop=True)
    df_out2 = df_out2[df_out2['yeardiff'] == 1]  # drops datapoints that are skip years

    output_file_name = 'data80.csv'
    #df_out2.to_csv(output_file_name, index=False)


