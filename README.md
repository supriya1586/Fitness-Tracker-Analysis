# Fitness-Tracker-Analysis

The primary goal of this study was to analyse fitness tracker data. The work was roughly divided in two parts
1.	Examining data and finding unusual behaviour patterns
2.	Studying machine learning models for the purpose of optimizing step count targets.

Overall, the key objectives can be summed up as following:
•	Analysing behaviour patterns of the Fitbit users by finding the correlation between plots
•	Using different methods for features selection that can be used for training machine learning models
•	Use different machine learning models to forecast the realistic goal in the form of daily total step count that can be achieved
•	Comparing the performances of different combinations of models and features selection methods for them

**Dataset**
This project used Fitbit records available publicly on Zenodo.org (Furberg et al., 2016) to analyse and correlate. The personal tracking data consists of thirty different users records over the duration of two months. During March 12th and May 12th, 2016, users participated in a distributed survey via Amazon Mechanical Turk and generated two monthly datasets, each consisting of eleven and eighteen csv files. Thirty Fitbit users participated to provide data from their personal trackers, which comprised of per minute output for physical activity, heart rate, and sleep tracking. The data set in question is provided anonymously by the users and might have come from all over the world. Unfortunately, not knowing gender, age, or location limited the scope of analysis that could be performed.


**Data Pre processing**
Records for days in which Fitbit was not worn were deleted. The resultant data was grouped by user id and date. Spreadsheets from both periods were concatenated and Ids were replaced by generic names. ‘ActivityDate’ column was converted to Datetime datatype. Empty cells were replaced by zeros.

**Analysis**
After pre-processing of the data, to study the data and to understand the correlation between the features different graphs were plotted.
Heart Rate Analysis 
On the above heart data within the resting time, mean of the heart rate counts were calculated grouped by Ids so that average resting heart rate can be obtained. With this data the bar graph is plotted which gives pretty clear understanding of average resting heart rate of all the users which can also highlight if there is anomalous behaviour.

**Weight data Analysis**
WeightPounds and BMI columns were taken into account for plotting weight information. Average weight is calculated for all the users.  Weight against BMI graph was plotted which gives information about average weight of individual against BMI.

![image](https://user-images.githubusercontent.com/87475754/143394103-4a331807-6a94-4ffd-aae6-42b663d60a19.png)


**Heart rate and weight data combined analysis**
Heart risk was also tried to be predicted by combining heart rate and weight records together. For this Ids who have records in both the sheets were picked and their corresponding information was merged in a single dataset. Unnecessary columns like 'WeightPounds','Fat','LogId', 'IsManualReport','Date' were dropped.  Records having value 0 for any remaining column were removed. Duplicate id records were removed. According to original objective of this study, heart risk was supposed to be predicted on the information if users do not have normal BMI which is 20 – 24 and normal resting heart rate which is 60 – 100 beats per minute. However, after cleaning the data only 5 proper records were left which were not sufficient to make any predictions. 

![image](https://user-images.githubusercontent.com/87475754/143394294-d6749cfa-3428-4c71-8b86-2d6181978c89.png)


**Sleep Records Analysis**
Sleep data has three important columns Id, Value and Date, of which Value column was not precisely readable since it has numerical value 1, 2 and 3. To make it more practical, new dataframe sleepDF was created with columns Name, date, and minsAsleep, minsRestless, minsAwake, and TotalMinsInBed. Values 1, 2 and 3 were categorized into minsAsleep, minsRestless, minsAwake respectively. For TotalMinsInBed all the minutes for the day were summed up. and data was grouped by date. This step reduced big sleep data of 387080 rows into 901 better organized rows.

QualitySleepRatio was calculated by MinsAsleep divided by TotalMinsInBed. With the help of which average quality sleep was calculated. This was used to plot top 5 best sleepers. Also, poor sleep quality ratio was calculated by using (MinsAwake + MinsRestless) divided by TotalMinsInBed.  This was used to plot top 5 worst sleepers. 
Some more graphs with TotalSteps against TotalDistance, TotalSteps against AllActiveMinutes, TotalSteps against QualitySleepRatio and AllActiveMinutes against QualitySleepRatio were also plotted to find the correlation between activities and sleep. 


**K-Means Application**
Sleep data was used for analysis with K means so that patterns can be observed.
Parameters 'TotalSteps' 'AllActiveMins','TotalDistance' were considered for plotting clusters so that we could predict the 'QualitySleepRatio' sleep ratio against them.

![image](https://user-images.githubusercontent.com/87475754/143394416-70c046f9-10f2-40a0-87f8-19b6b2bdc0c1.png)

**Calorie Burn Correlation plot**
![image](https://user-images.githubusercontent.com/87475754/143394515-634b819c-f098-4091-8893-05be8a9e2d21.png)


**Development of model**
Different machine learning models were used to predict total steps for a daily achievable target for users. These models were chosen based on their compatibility and ability. For numeric and categorical predictors, as well as classification and regression tasks, random forest models are adaptable. Random forest models are also easier to interpret and less prone to underfitting. Linear regression is more straightforward to implement, analyse, and train. When compared to certain others, it has a significantly lesser time complexity. Because of its regularisation penalty, Lasso regression is resistant to overfitting. XGBoost takes advantage of parallel processing, which is why it is so much faster.  
Above models were implemented in combination with below four cases.
1.	Using Recursive Features Elimination for features selection
2.	Using Select K best method for features selection
3.	Using new features formed by Principal Component Analysis
4.	Using all features in dataset


