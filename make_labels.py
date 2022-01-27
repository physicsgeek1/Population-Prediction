import pandas as pd
import numpy as np
#import sklearn
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.style.use('seaborn-white')
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('../../Weather_api/Final2.csv')
    df = df[df['Station_ID'].notnull()].sort_values(by=['id','year'])
    #df['popchange'] = df['pop'].diff()
    #df['popchange'] = df['popchange'].fillna(df.groupby('id')['popchange'].transform('mean'))
    
    b = np.array([df['id']]).transpose()
    pop = np.array([df['pop']]).transpose()
    #popchange = np.array([df['popchange']]).transpose()
    uid = np.unique(b)
    idpop = np.append(b,pop, axis=1)
    #idpop = np.append(idpop,popchange, axis=1)
    stdscale = []
    mmscale = []
#    stdchange = []
#    mmchange = []
    ss = StandardScaler()
    mm = MinMaxScaler()
    
    for ids in uid:
        temp = idpop[idpop[:,0]==ids]
        temp2 = temp[:,1]
        #temp3 = temp[:,2]
        if len(stdscale) == 0:
            stdscale = ss.fit_transform(temp2.reshape(-1,1))
            #stdchange = ss.fit_transform(temp3.reshape(-1,1))
        else:
            ssscale = ss.fit_transform(temp2.reshape(-1,1))
            stdscale = np.append(stdscale, ssscale, axis=0)
            #sschange = ss.fit_transform(temp3.reshape(-1,1))
            #stdchange = np.append(stdchange, sschange, axis=0)
            del ssscale
#            del sschange
            
        if len(mmscale) == 0:
            mmscale = mm.fit_transform(temp2.reshape(-1,1))
            #mmchange = mm.fit_transform(temp3.reshape(-1,1))
        else:
            mscale = mm.fit_transform(temp2.reshape(-1,1))
            mmscale = np.append(mmscale, mscale, axis=0)
            #mchange = mm.fit_transform(temp3.reshape(-1,1))
            #mmchange = np.append(mmchange, mchange, axis=0)            
            del mscale  
#            del mchange
#        del temp
#        del temp2
#        del temp3
        
    df['stdscale'] = stdscale
    df['mmscale'] = mmscale

#    df['stdchange'] = stdchange
#    df['mmchange'] = mmchange

    df1 = df.loc[:, df.isnull().mean() <= 0.3]  
    X = df1.isnull().mean()

    df1['TMAX'] = df1['TMAX'].fillna(df1.groupby('Station_ID')['TMAX'].transform('mean'))  
    df1['DP01'] = df1['DP01'].fillna(df1.groupby('Station_ID')['DP01'].transform('mean'))  
    df1['DP10'] = df1['DP10'].fillna(df1.groupby('Station_ID')['DP10'].transform('mean'))  
    df1['DP1X'] = df1['DP1X'].fillna(df1.groupby('Station_ID')['DP1X'].transform('mean'))  
    df1['DSNW'] = df1['DSNW'].fillna(df1.groupby('Station_ID')['DSNW'].transform('mean'))  
    df1['DT00'] = df1['DT00'].fillna(df1.groupby('Station_ID')['DT00'].transform('mean'))  
    df1['DT32'] = df1['DT32'].fillna(df1.groupby('Station_ID')['DT32'].transform('mean'))  
    df1['DX70'] = df1['DX70'].fillna(df1.groupby('Station_ID')['DX70'].transform('mean'))  
    df1['DX90'] = df1['DX90'].fillna(df1.groupby('Station_ID')['DX90'].transform('mean'))  
    df1['EMNT'] = df1['EMNT'].fillna(df1.groupby('Station_ID')['EMNT'].transform('mean'))  
    df1['EMSN'] = df1['EMSN'].fillna(df1.groupby('Station_ID')['EMSN'].transform('mean'))  
    df1['EMXP'] = df1['EMXP'].fillna(df1.groupby('Station_ID')['EMXP'].transform('mean'))  
    df1['EMXT'] = df1['EMXT'].fillna(df1.groupby('Station_ID')['EMXT'].transform('mean'))  
    df1['PRCP'] = df1['PRCP'].fillna(df1.groupby('Station_ID')['PRCP'].transform('mean'))  
    df1['SNOW'] = df1['SNOW'].fillna(df1.groupby('Station_ID')['SNOW'].transform('mean'))  
    df1['TMIN'] = df1['TMIN'].fillna(df1.groupby('Station_ID')['TMIN'].transform('mean'))  

    df2 = df1.dropna(axis = 0)
    
    df3 = df.loc[:, df.isnull().mean() <= 0.2]  
    Y = df3.isnull().mean()

    df3['DP01'] = df3['DP01'].fillna(df1.groupby('Station_ID')['DP01'].transform('mean'))  
    df3['DP10'] = df3['DP10'].fillna(df1.groupby('Station_ID')['DP10'].transform('mean'))  
    df3['DP1X'] = df3['DP1X'].fillna(df1.groupby('Station_ID')['DP1X'].transform('mean'))  
    df3['EMXP'] = df3['EMXP'].fillna(df1.groupby('Station_ID')['EMXP'].transform('mean'))  
    df3['PRCP'] = df3['PRCP'].fillna(df1.groupby('Station_ID')['PRCP'].transform('mean')) 
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
    
#    fig, axs = plt.subplots(1,2)
#    fig.suptitle('Populations')
#    axs[0,0].plot(df2['year'], df2['pop'])
#    axs[0,0].set_title('Populations70')
#    axs[0,1].plot(df4['year'], df4['pop'])
#    axs[0,1].set_title('Populations80')
#    plt.xlabel('Year')
#    plt.ylabel('Scaled Population')


    ######### groupby id, calculate percent change in pop, and remove any data that isnt an incremental year
    grouped = df2.groupby(['id']).apply(lambda x: x.sort_values(by=['year']))
    grouped['yeardiff'] = grouped['year'].diff()
    grouped['stddiff'] = grouped['stdscale'].diff()  # pct = percent, this gives results in decimal and not %
    grouped['mmdiff'] = grouped['mmscale'].diff()  # pct = percent, this gives results in decimal and not %
    grouped['lastpop'] = grouped['mmscale'].shift(1)
    df_out = grouped.reset_index(drop=True)
    df_out = df_out[df_out['yeardiff'] == 1]  # drops datapoints that are skip years

    output_file_name = '../../processed/data70.csv'
    #df_out.to_csv(output_file_name, index=False)
    print(df_out.shape)

    grouped2 = df4.groupby(['id']).apply(lambda x: x.sort_values(by=['year']))
    grouped2['yeardiff'] = grouped2['year'].diff()
    grouped2['stddiff'] = grouped2['stdscale'].diff()  # pct = percent, this gives results in decimal and not %
    grouped2['mmdiff'] = grouped2['mmscale'].diff()  # pct = percent, this gives results in decimal and not %
    grouped2['lastpop'] = grouped2['mmscale'].shift(1)
    df_out2 = grouped2.reset_index(drop=True)
    df_out2 = df_out2[df_out2['yeardiff'] == 1]  # drops datapoints that are skip years

    output_file_name = '../../processed/data80.csv'
    #df_out2.to_csv(output_file_name, index=False)
    print(df_out2.shape)

    #percent of filled feature values by year
   # y = df.loc[:, df.isnull().mean() <= 0.3]
    y = df.drop(['Class','Order','Family','Genus','Species','Common Name','Region',
              'Are coordinates for specific location?','system','biome','realm',
              'Native','Alien','Invasive','Units','Sampling method','Data transformed'], axis=1)
    y = y.drop(['Station_ID'], axis=1)
    pctfill = y.groupby('year').apply(lambda x: x.notnull().sum()/len(x)*100)

#df_scaled = pd.DataFrame(ss.fit_transform(df),columns = df.columns)
    #scaler.fit_transform(dfTest[['A','B']].values)