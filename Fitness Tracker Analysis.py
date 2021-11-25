#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import math
from itertools import chain
from collections.abc import Iterable
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Get March-April data for daily activity of users
dailyActivity1 = pd.read_csv('data/Fitabase Data 3.12.16-4.11.16/dailyActivity_merged.csv')
print('Daily Activity March-April records: ' + str(len(dailyActivity1)) +
     ', number of columns: ' + str(len(dailyActivity1.columns)))

# Get April-May data for daily activity of users
dailyActivity2 = pd.read_csv('data/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv')
print('Daily Activity April-May records: ' + str(len(dailyActivity2)) 
      +
     ', number of columns: ' + str(len(dailyActivity2.columns)))

# concat the two data frames
dailyActivity = pd.concat([dailyActivity1, dailyActivity2])
print('Total number of records: ' + str(len(dailyActivity)) +
     ', number of columns: ' + str(len(dailyActivity.columns)))

# set the index to the user ID
dailyActivity.set_index('Id', inplace=True)

# get rid of empty cells by replacing them with zero
dailyActivity = dailyActivity.fillna(0)

# make the data strings actual dates
dailyActivity['ActivityDate'] = pd.to_datetime(dailyActivity['ActivityDate'], format='%m/%d/%Y')

#copying dataset before masking ids to use it as it is for machine learning study
dailyActivity_orig = dailyActivity.copy()

# Establish 35 short generic names to replace the numeric anonymous id.
# It will look more readable in the charts.
genericNames = ['Amy','Becky','Charlie','Dave','Emily','Fran','George',
                'Harry','Iris','Jenny','Kyle','Leo','Mary','Nina','Oscar',
                'Pete','Quincey','Rachel','Sam','Tim','Una','Vicky','Wes',
                'Xavier','Yvonne','Zoe','Adam','Ben','Cathy','Donna','Eric',
                'Fred','Grace','Heather','Ian']

uniqueUserIdList = dailyActivity.index.unique().sort_values().values
idx = 0
for uniqueId in uniqueUserIdList:
    dailyActivity.rename(index={uniqueId: genericNames[idx]}, inplace=True)
    idx+=1
 

dailyActivity.head()

# Get March-April data for sleep records of users
sleepM1 = pd.read_csv('data/Fitabase Data 3.12.16-4.11.16/minuteSleep_merged.csv')
print('Sleep (minutes) March-April records: ' + str(len(sleepM1)) +
     ', number of columns: ' + str(len(sleepM1.columns)))

# Get April-May data for sleep records of users
sleepM2 = pd.read_csv('data/Fitabase Data 4.12.16-5.12.16/minuteSleep_merged.csv')
print('Sleep (minutes) April-May records: ' + str(len(sleepM2)) +
     ', number of columns: ' + str(len(sleepM2.columns)))

# concat the two data frames
sleepM = pd.concat([sleepM1, sleepM2])

# set the index to the user ID
sleepM.set_index('Id', inplace=True)

# rename the unique ids to human-readable names
idx = 0
for uniqueId in uniqueUserIdList:
    sleepM.rename(index={uniqueId: genericNames[idx]}, inplace=True)
    idx+=1

# make the 'date' column an actual datetime type, and just keep the date,
# drop the hour/min/second
sleepM['date'] = pd.to_datetime(sleepM['date']).dt.date

# delete logId - we don't need it
del sleepM['logId']

print('Total Sleep (minutes) records: ' + str(len(sleepM)) +
     ', number of columns: ' + str(len(sleepM.columns)))
sleepM.head()


# In[2]:


# Get March-April data for weight records of users
weightLog1 = pd.read_csv('data/Fitabase Data 3.12.16-4.11.16/weightLogInfo_merged.csv')
print('Weight Log March-April records: ' + str(len(weightLog1)) +
     ', number of columns: ' + str(len(weightLog1.columns)))

# Get April-May data for weight records of users
weightLog2 = pd.read_csv('data/Fitabase Data 4.12.16-5.12.16/weightLogInfo_merged.csv')
print('Weight Log April-May records: ' + str(len(weightLog2)) +
     ', number of columns: ' + str(len(weightLog2.columns)))

# concat the two data frames
weightLog = pd.concat([weightLog1, weightLog2])
weightLog_orig = weightLog.copy(deep=True)
weightLog.set_index('Id', inplace=True)


# In[3]:



# rename the unique ids to human-readable names
idx = 0
for uniqueId in uniqueUserIdList:
   weightLog.rename(index={uniqueId: genericNames[idx]}, inplace=True)
   idx+=1

print('Total Weight Log records: ' + str(len(weightLog)) +
    ', number of columns: ' + str(len(weightLog.columns)))
weightLog.head()


# In[4]:


# Get March-April data for heart rate records of users
heartrate1 = pd.read_csv('data/Fitabase Data 3.12.16-4.11.16/heartrate_seconds_merged.csv', index_col='Time', parse_dates=True)
print('Heart Rate March-April records: ' + str(len(heartrate1)) +
     ', number of columns: ' + str(len(heartrate1.columns)))

# Get April-May data for heart rate records of users
heartrate2 = pd.read_csv('data/Fitabase Data 4.12.16-5.12.16/heartrate_seconds_merged.csv', index_col='Time', parse_dates=True)
print('Heart Rate April-May records: ' + str(len(heartrate2)) +
     ', number of columns: ' + str(len(heartrate2.columns)))


# concat the two data frames
heartrate = pd.concat([heartrate1, heartrate2])



print('Total Heart Rate records: ' + str(len(heartrate)) +
     ', number of columns: ' + str(len(heartrate.columns)))
heartrate.head()


# In[5]:


heartrate = heartrate.between_time('00:00:00','08:00:00')

heartratedf = heartrate.reset_index()

# set the index to the user ID
df = heartratedf.set_index('Id')


uniqueUserIdList = df.index.unique().sort_values().values
#uniqueUserIdList
idx = 0

# rename the unique ids to human-readable names
for uniqueId in uniqueUserIdList:
    df.rename(index={uniqueId: genericNames[idx]}, inplace=True)
    idx+=1
    
    
df['Time'] = pd.to_datetime(df['Time']).dt.date

HRDCounts = df.groupby(['Id']).mean()

HRDCounts.shape[0]

df = HRDCounts.sort_values(by='Id',                                               ascending=True)

    
print(df.shape[0])


# In[6]:


thePlot = df.plot(kind='bar', stacked=True, figsize=(12,10), fontsize=12)
thePlot.set_title('Average resting heart rate of all users')
thePlot.set_xlabel('Users')
thePlot.get_legend().remove()
thePlot.set_ylabel('Average Resting Heart Rate') 


# In[7]:


# merging weight and heart rate records using ids  
merged_df = weightLog_orig.merge(heartratedf, how="outer")
merged_df


# In[8]:


merged_df = merged_df.fillna(0)
#dropped all other unrelated columns
merged_df.drop(['WeightPounds','Fat','LogId', 'IsManualReport','Date'], axis=1, inplace=True)
merged_df = merged_df[(merged_df[['WeightKg','BMI','Value' ]] != 0).all(axis=1)]
merged_df.groupby('Id')
merged_df.drop_duplicates(subset ="Id",
                     keep = "first", inplace = True)
 
merged_df


# In[9]:


# create a function to plot a name
def plotSteps(userId, thePlot):
    data = dailyActivity.loc[userId].sort_values(by='ActivityDate')
    #print(data)
    thePlot.plot(data.ActivityDate, data.TotalSteps)

# find top steppers based on total steps
def getTopSteppers(limit=10):
    # Only grab the total steps column and reset the index
    topSteppers = dailyActivity['TotalSteps'].reset_index()
    topSteppers = topSteppers.groupby('Id').sum()          .sort_values(by='TotalSteps', ascending=False)
    return topSteppers.head(limit)

# create a plot of top steppers
def plotTopSteppers(limit=5):
    fig = plt.figure(figsize=(12,10))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax.set_title('Top ' + str(limit) + ' Steppers')
    topSteppers = getTopSteppers(limit)
    for stepperId in topSteppers.index:
        plotSteps(stepperId, ax)     
    ax.legend(topSteppers.index.values, loc='best') 
    ax.set_xlabel('Date')
    ax.set_ylabel('Steps')
  

plotTopSteppers()


# In[10]:


# find worst steppers based on total steps
def getWorstSteppers(limit=10):
    # Only grab the total steps column and reset the index
    worstSteppers = dailyActivity['TotalSteps'].reset_index()
    worstSteppers = worstSteppers.groupby('Id').sum()          .sort_values(by='TotalSteps', ascending=False)
    return worstSteppers.tail(limit)

# create a plot of worst steppers
def plotWorstSteppers(limit=5):
    fig = plt.figure(figsize=(12,10))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax.set_title('Top ' + str(limit) + ' Worst Steppers')
    worstSteppers = getWorstSteppers(limit)
    for stepperId in worstSteppers.index:
        plotSteps(stepperId, ax)     
    ax.legend(worstSteppers.index.values, loc='best') 
    ax.set_xlabel('Date')
    ax.set_ylabel('Steps')

plotWorstSteppers()


# In[11]:


# calculate active and sedentary minute totals
def getAllActiveMinutes(df):
    activeMinsInfo = df.loc[:,['VeryActiveMinutes','FairlyActiveMinutes',
                              'LightlyActiveMinutes','SedentaryMinutes']] \
                                .reset_index()
    
    activeMinsInfo = activeMinsInfo.groupby('Id').sum()
    
    activeMinsInfo['AllActiveMins'] = activeMinsInfo['VeryActiveMinutes'] +         activeMinsInfo['FairlyActiveMinutes'] + activeMinsInfo['LightlyActiveMinutes']
    return activeMinsInfo
    
# function to plot most active data
def plotMostActive(df, limit=10):
    df = getAllActiveMinutes(df).sort_values(by='AllActiveMins', ascending=False)[:limit]   
    df.drop(['AllActiveMins','SedentaryMinutes'], axis=1, inplace=True)
    thePlot = df.plot(kind='barh', stacked=True, figsize=(12,10), fontsize=14)
    thePlot.set_title('Top ' + str(limit) + ' Most Active Users')
    thePlot.set_ylabel('Users')
    thePlot.set_xlabel('Total Overall Minutes')
 

plotMostActive(dailyActivity)   


# In[12]:


# function to plot least active data
def plotMostSedentary(df, limit=10):
    df = getAllActiveMinutes(df).sort_values(by='SedentaryMinutes', ascending=False)[:limit]   
    del df['AllActiveMins'] 
    # move Sedentary Minutes to the first position to make the chart look nice
    df = df[['SedentaryMinutes','LightlyActiveMinutes',
             'FairlyActiveMinutes','VeryActiveMinutes']]
    thePlot = df.plot(kind='barh', stacked=True, figsize=(12,10), fontsize=14)
    thePlot.set_title('Top ' + str(limit) + ' Most Sedentary Users')
    thePlot.set_ylabel('Users')
    thePlot.set_xlabel('Total Overall Minutes')
    
plotMostSedentary(dailyActivity)  


# In[13]:


#plotting all types of distances covered
distance = ['VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance']
dailyActivity[distance].plot(kind='box', figsize=(12,5));


# In[14]:


#plot average distances by all users with intesities
d = dailyActivity.groupby('Id')[distance].mean().sort_values(by='VeryActiveDistance')
d.plot(kind='bar', stacked=True, figsize=(10,5))
plt.grid(axis='y', alpha=0.3)
plt.ylabel('Distance Travelled')
plt.xlabel('Users')
plt.title('Average distances by users');


# In[15]:


#plot average active time by day of the week
week = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
dailyActivity['WeekDay'] = dailyActivity['ActivityDate'].dt.dayofweek
day = dailyActivity.groupby('WeekDay')[['VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes']].mean()
(day/60).plot(kind='bar', stacked=True, figsize=(11,9))
plt.grid(axis='y', alpha=0.5)
plt.title('Average Time by Day of the Week ')
plt.ylabel('hours')
plt.xlabel(xlabel=None)
plt.xticks(ticks = [0,1,2,3,4,5,6], labels=week)
plt.legend();


# In[16]:


# calculate active and sedentary minute totals
def getAllActiveDistance(df):
    activeDistInfo = df.loc[:,['TotalDistance','VeryActiveDistance', 
                              'ModeratelyActiveDistance',
                              'LightActiveDistance','SedentaryActiveDistance']] \
                                .reset_index()
    activeDistInfo = activeDistInfo.groupby('Id').sum()

    return activeDistInfo
    
# function to plot most active data
def plotMostActiveDistance(df, limit=10):
    df = getAllActiveDistance(df).sort_values(by='TotalDistance', ascending=False)[:limit]       
    del df['TotalDistance']
    thePlot = df.plot(kind='bar', stacked=True, figsize=(12,10), fontsize=14)
    thePlot.set_title('Top ' + str(limit) + ' Most Far-Roaming Users')
    thePlot.set_xlabel('Users')
    thePlot.set_ylabel('Total Overall Distance') 
    
plotMostActiveDistance(dailyActivity)

# function to plot least distance data
def plotLeastActiveDistance(df, limit=10):
    df = getAllActiveDistance(df).sort_values(by='TotalDistance',                                               ascending=True)[:limit]       
    del df['TotalDistance']
    thePlot = df.plot(kind='bar', stacked=True, figsize=(12,10), fontsize=14)
    thePlot.set_title('Top ' + str(limit) + ' Least Far-Roaming Users')
    thePlot.set_xlabel('Users')
    thePlot.set_ylabel('Total Overall Distance') 
    
plotLeastActiveDistance(dailyActivity)


# In[17]:


# first get rid of the columns we don't need
def displayWeightInfo(df):
    weight = df.drop(['Date','WeightKg','Fat','IsManualReport','LogId'], axis=1)
    avgWeight = weight.groupby('Id').mean() 
    
    fig, ax = plt.subplots() # plt.figure(figsize=(12,10))
    fig.set_figheight(14)
    fig.set_figwidth(10)
    ax.scatter(avgWeight['WeightPounds'], avgWeight['BMI'],
                          s=3000, c=avgWeight['BMI'], cmap="Greens", 
                          alpha=0.4, edgecolors="grey", linewidth=2)
    # annotate each bubble with the user name
    for item in avgWeight.index:
        ax.annotate(item, (avgWeight.loc[item]['WeightPounds'], 
                          avgWeight.loc[item]['BMI']))
        
    # Add titles (main and on axis)
    ax.set_xlabel("Weight (pounds)")
    ax.set_ylabel("BMI")
    ax.set_title("Weight vs BMI")
    
displayWeightInfo(weightLog)


# In[18]:


# Put the sleep info into a more manageable data frame based on asleep,
# restless, and awake counts per day instead of by minute

sleepCounts = sleepM.groupby(['Id','value','date']).date.count()
sleepCountsByValue = sleepCounts.unstack().fillna(0)

sleepCountsByValue.head()

def getSleepCountDF(df, userList) :
    # create a new data frame with Id, date, and minsAsleep (value = 1), 
    # minsRestless (value = 2), minsAwake (value = 3), and TotalMinsInBed
    sleepDF = pd.DataFrame(columns=['Name','Date','MinsAsleep','MinsRestless',
                                    'MinsAwake','TotalMinsInBed'])
    df.reset_index()
    df.head();
    for name in userList:
        if name in df.index:
            userRows = df.loc[[(name)]]
            for day in userRows.columns:
                currVals = processDay(df, name, day)
                # No point in adding a day where sleep wasn't tracked
                if currVals['TotalMinsInBed'] > 0 : 
                    sleepDF = sleepDF.append(currVals, ignore_index=True, 
                                             sort=False)
    print("We just reduced " + str(len(sleepM)) + " sleepM rows to " +          str(len(sleepDF)) + " better organized sleepDF rows.")
    sleepDF = sleepDF.set_index('Name')
    return sleepDF

def processDay(df, name, day) :
    if (name, 1) in df.index:
        minsAsleep = df.loc[[(name, 1)],[day]].values[0][0]
    else:
        minsAsleep = 0
        
    if (name, 2) in df.index:
        minsRestless = df.loc[[(name, 2)],[day]].values[0][0]
    else: 
        minsRestless = 0
        
    if (name, 3) in df.index:
        minsAwake = df.loc[[(name, 3)],[day]].values[0][0]
    else:
        minsAwake = 0
        
    totalMinsInBed = minsAsleep + minsRestless + minsAwake
    
    return {'Name': name, 'Date': day, 'MinsAsleep': minsAsleep, 
                'MinsRestless': minsRestless, 'MinsAwake': minsAwake,
                'TotalMinsInBed': totalMinsInBed}


sleepDF = getSleepCountDF(sleepCountsByValue, genericNames)
sleepDF.head()


# In[19]:


# find the users with highest MinsAsleep-to-TotalMinsInBed

def getBestSleepers(sleepdf, limit=5):
    df = sleepdf.copy()
    df['QualitySleepRatio'] = df['MinsAsleep'] / df['TotalMinsInBed'] 
    avgQualitySleep = df.groupby('Name').mean()         .sort_values(by='QualitySleepRatio', ascending=False)[:limit]
    return avgQualitySleep
    
    
bestSleepers = getBestSleepers(sleepDF, 5)
bestSleepers


# In[20]:


def plotBestSleepers(df):    
    plt.figure(figsize=(12,10))
    plt.stackplot(df.index, df['MinsAsleep'].values,
                 df['MinsRestless'].values, df['MinsAwake'].values)
    plt.legend(df.columns.values)
    plt.xlabel('Users')
    plt.ylabel('Total Sleep Minutes') 
    plt.title('Top 5 best sleepers')

plotBestSleepers(bestSleepers)


# In[21]:


# find the users with highest MinsAwake/MinsRestless-to-TotalMinsInBed

def getWorstSleepers(sleepdf, limit=5) :
    df = sleepdf.copy()
    df['PoorSleepRatio'] = (df['MinsAwake'] + df['MinsRestless'] )                 / df['TotalMinsInBed'] 
    avgPoorSleep = df.groupby('Name').mean()         .sort_values(by='PoorSleepRatio', ascending=False)[:limit]
    return avgPoorSleep
    
    
worstSleepers = getWorstSleepers(sleepDF, 10)
worstSleepers.head()
def plotWorstSleepers(df):    
    plt.figure(figsize=(12,10))
    plt.stackplot(df.index, df['MinsAsleep'].values,
                 df['MinsRestless'].values, df['MinsAwake'].values)
    plt.legend(df.columns.values)
    plt.xlabel('Users')
    plt.ylabel('Total Sleep Minutes') 
    plt.title('Top 5 worst sleepers')

plotWorstSleepers(worstSleepers)


# In[22]:



# make a master data frame that contains steps, active minutes, distance,
# and sleep
limit = 40 # we have under 40 people in our dataset
masterDF = pd.DataFrame() #index=dailyActivity.index, 
                        #columns=['TotalSteps','ActiveMins','Distance','Sleep'])
masterDF = masterDF.append(getTopSteppers(limit))
activeMins = pd.DataFrame(getAllActiveMinutes(dailyActivity)['AllActiveMins'])
masterDF = masterDF.join(activeMins)
activeDistance = pd.DataFrame(getAllActiveDistance(dailyActivity)['TotalDistance'])
masterDF = masterDF.join(activeDistance)
sleepInfo = pd.DataFrame(getBestSleepers(sleepDF, limit)['QualitySleepRatio'])
masterDF = masterDF.join(sleepInfo)

masterDF.sort_values('Id').head(limit)
plt.plot(masterDF['TotalSteps'], masterDF['TotalDistance'])
plt.title('TotalSteps vs TotalDistance')  
plt.xlabel('TotalSteps')
plt.ylabel('TotalDistance (miles)')


# In[23]:


#plot TotalSteps vs AllActiveMins
plt.plot(masterDF['TotalSteps'], masterDF['AllActiveMins'])
plt.title('TotalSteps vs AllActiveMins')  
plt.xlabel('TotalSteps')
plt.ylabel('AllActiveMins')


# In[24]:


#plot TotalSteps vs QualitySleepRatio

# drop NAs from sleep data for this part
masterDF = masterDF.dropna()

plt.plot(masterDF['TotalSteps'], masterDF['QualitySleepRatio'])
plt.title('TotalSteps vs QualitySleepRatio')  
plt.xlabel('TotalSteps')
plt.ylabel('QualitySleepRatio (% of TotalSleep)')


# In[25]:


# plot AllActiveMins vs QualitySleepRatio
plt.plot(masterDF['AllActiveMins'], masterDF['QualitySleepRatio'])
plt.title('AllActiveMins vs QualitySleepRatio')  
plt.xlabel('AllActiveMins')
plt.ylabel('QualitySleepRatio (% of TotalSleep)')


# In[30]:


import seaborn as sns
import numpy as np

def plot_heatmap(corrmat, correlationOf, title, darkTheme=False):
    if darkTheme:
        sns.set(style='darkgrid', palette='deep') # Using Seaborn for making heatmap
        cmap="YlGnBu"
    else:     
        sns.set(style = "white")
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(10, 10))
    hm = sns.heatmap(corrmat, mask=mask, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, cmap=cmap)
    hm.set_title(title)
    plt.yticks(rotation=0)
    plt.show()


# In[31]:


correlationOf = 'Calories Burned'
corrdf_calories = dailyActivity_orig[['Calories','TotalSteps', 'TotalDistance', 'SedentaryMinutes', 'FairlyActiveMinutes', 'VeryActiveMinutes', 'LightlyActiveMinutes']]
plot_heatmap(corrdf_calories.corr(), correlationOf, '')


# In[32]:


# find number of K to select in K-means

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans

Sum_of_squared_distances = []
K = range(1,10)
for num_clusters in K :
 kmeans = KMeans(n_clusters=num_clusters)
 kmeans.fit(masterDF)
 Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[55]:


# now make k-means cluster with the data.  We can cluster on all four
# columns, but we can only visualize three

import numpy as np
from sklearn import cluster, decomposition
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

def makeClusters(df, x, y, z, sleep, numClusters, normalize=False):
    dataset = df[[x,y,z, sleep]].copy()
    print(dataset.head())
    datasetArray = dataset.values
    data = np.array(list(datasetArray), dtype=np.float64)

    if normalize:
         
        data = (data - data.mean(axis=0)) / data.std(axis=0)

    kmeans = cluster.KMeans(n_clusters=numClusters, n_init=15)
    kmeans.fit(data)
    
    colors = 'rbyg'  
    fig = plt.figure(figsize=(18,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], marker='o',  
               c=[colors[g] for g in kmeans.labels_], alpha=0.7, 
               s=40, linewidth=1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title('K-Means with # Clusters = ' + str(numClusters))
    grid_wt = 0.5
    ax.w_xaxis._axinfo.update({'grid' : {'color': (0, 0, 0, grid_wt),
                                         'linewidth': 1,
                                         'linestyle':'solid'}})
    ax.w_yaxis._axinfo.update({'grid' : {'color': (0, 0, 0, grid_wt),
                                         'linewidth': 1,
                                         'linestyle':'solid'}})
    ax.w_zaxis._axinfo.update({'grid' : {'color': (0, 0, 0, grid_wt),
                                         'linewidth': 1,
                                         'linestyle':'solid'}})
    
    # annotate each dot with the user name    
    for i in range(len(df.index)):
        x2, y2, _ = proj3d.proj_transform(data[i][0],data[i][1],
                                              data[i][2], ax.get_proj())
        label = ax.annotate(
            df.index.values[i], 
            xy = (x2, y2), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
           
    plt.grid(lw=2)
    plt.show()
makeClusters(masterDF, 'TotalSteps','AllActiveMins','TotalDistance',            'QualitySleepRatio', numClusters=3, normalize=True)


# In[34]:


# implementing other machine learning models on daily activity data to predict total steps that can be set as goal
import datetime as dt
import numpy as np

df_date_encoding = dailyActivity_orig.copy()

df_date_encoding['year'] = df_date_encoding['ActivityDate'].dt.year
df_date_encoding['month'] = df_date_encoding['ActivityDate'].dt.month
df_date_encoding['day_of_year'] = df_date_encoding['ActivityDate'].dt.day

df_date_encoding.drop(['ActivityDate'], axis=1, inplace =True)

df_date_encoding.head()


# In[35]:


from sklearn.metrics import mean_absolute_error, r2_score,mean_absolute_percentage_error
# print all the results
def print_results(name, test_labels, predictions):
    print("results for ", name)
    print("Mean absolute error: %.2f"% mean_absolute_error(test_labels, predictions))
    print("r2 score: ",r2_score(test_labels, predictions))
    print("mean absolute percentage error", mean_absolute_percentage_error(test_labels, predictions))
    print('average daily steps that can be targetted ',predictions.mean())


# In[36]:


import numpy as np
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE 
from sklearn.preprocessing import MinMaxScaler

# get the dataset
x = df_date_encoding.drop(labels='TotalSteps', axis=1).values
y = df_date_encoding['TotalSteps'].values
 
#array  = df_date_encoding.values
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(x,y)

# function to split dataset with various test size
def splitDataset(selected, test_size):
    data = df_date_encoding[selected].values
    
     
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(data, y, test_size = test_size, random_state = 42)
    return train_features, test_features, train_labels, test_labels

# common function for fitting data to all models
def model_fitting(model,train_features,train_labels, test_features, test_labels, name ):    
    
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    print_results(name, test_labels, predictions)
    return predictions
    
estimators = LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor()

# function to implement all models, gather results and plot the predictions of all the algorithms 
def model_implementation(train_features_m, train_labels_m,test_features_m, test_labels_m):
    results  = list() 
    model = RandomForestRegressor(n_estimators = 700, random_state = 42)    
    results.append(model_fitting(model, train_features_m,train_labels_m, test_features_m, test_labels_m, 'RandomForestRegressor'))
   
    model = LinearRegression()     
    results.append(model_fitting(model, train_features_m,train_labels_m, test_features_m, test_labels_m, 'LinearRegression'))
    
   
    model = Lasso(alpha= 1.0)
    results.append(model_fitting(model, train_features_m,train_labels_m, test_features_m, test_labels_m, 'Lasso'))
   
   
    model = XGBRegressor()
    results.append(model_fitting(model, train_features_m,train_labels_m, test_features_m, test_labels_m, 'XGBRegressor'))
  
    model = DecisionTreeRegressor()
    results.append(model_fitting(model, train_features_m,train_labels_m, test_features_m, test_labels_m, 'DecisionTreeRegressor'))
   
    names = 'RandomForestRegressor','LinearRegression', 'Lasso', 'XGBRegressor', 'DecisionTreeRegressor'
    fig, ax = plt.subplots(figsize=(12, 7))
    box = ax.boxplot(results, labels=names, patch_artist=True)  
    #colors = ['pink', 'lightblue', 'lightgreen', 'coral' ,'navajowhite']
    #colors = ['springgreen', 'paleturquoise','limegreen', 'aquamarine','lightgreen']    
    #colors =['aqua','deepskyblue', 'cornflowerblue', 'lightblue','steelblue']
    colors = ['lightyellow', 'mistyrose', 'honeydew', 'aliceblue', 'lavender']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.show()
  


# In[37]:


# recursive feature elimination for features selection
for estimator_reg in estimators: 
    test_sizes = 0.25, 0.3
    print("for estimator ", estimator_reg)
        
    rfe = RFE(estimator=estimator_reg, n_features_to_select=4)
    rfe.fit(X, y)
    df_rank = pd.DataFrame(rfe.support_,index =df_date_encoding.drop(labels='TotalSteps', axis=1).columns, columns=['Rank'])
    df_rank_filtered = df_rank.loc[df_rank['Rank'] == True]
    index = df_rank_filtered.index
    selected = index.tolist()
    print("features selected: ",selected)
    for test_size in test_sizes:
        print("for test_size: ", test_size)
        train_features, test_features, train_labels, test_labels = splitDataset(selected, test_size)
        model_implementation(train_features, train_labels,test_features, test_labels)
        


# In[38]:


# PCA for features selection

test_sizes = 0.25, 0.3

for test_size in test_sizes:
    print("for test_size: ", test_size)
    X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test = train_test_split(df_date_encoding[df_date_encoding.columns[df_date_encoding.columns != 'TotalSteps']],
                   df_date_encoding['TotalSteps'], test_size= test_size, random_state=1)
    
    # Standardize the dataset; This is very important before you apply PCA
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Perform PCA

    from sklearn.decomposition import PCA
    pca = PCA()
    
    # Determine transformed features
    
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    print(pca.explained_variance_ratio_.cumsum())


    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    model_implementation(X_train_pca, y_train,X_test_pca, y_test)


# In[39]:


# select K best method for feature selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression

test_sizes = 0.25, 0.3

for test_size in test_sizes:
    print("for test_size: ", test_size)
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= test_size, random_state=1)
    # feature selection

    score_fun = f_regression, mutual_info_regression

    for score_func in score_fun:
        print("for score_func: ", score_func)
        fs = SelectKBest(score_func=score_func, k=10) 

        fs.fit(X_train, y_train)    
        X_train_fs = fs.transform(X_train) 
        X_test_fs = fs.transform(X_test)
        model_implementation(X_train_fs, y_train,X_test_fs, y_test)
    


# In[40]:



# for all features selected
test_sizes = 0.25, 0.3

for test_size in test_sizes:
    print("for test_size: ", test_size)
    X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test = train_test_split(df_date_encoding[df_date_encoding.columns[df_date_encoding.columns != 'TotalSteps']],
                   df_date_encoding['TotalSteps'], test_size= test_size, random_state=1)
     
    model_implementation(X_train, y_train,X_test, y_test)


# In[49]:


import xgboost as xgb
RegModel=XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=500, objective='reg:squarederror', booster='gbtree')

XGB=RegModel.fit(train_features,train_labels)





# In[50]:


xgb.plot_tree(RegModel,num_trees=3)
plt.rcParams['figure.figsize'] = [50, 30]
plt.show()


# In[ ]:




