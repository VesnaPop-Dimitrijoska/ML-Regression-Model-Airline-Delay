# 
# Project Title:
Machine Learning Regression Model on Airline Delay dataset.
#
# Table of Contents:

  1. Data Cleaning
  2. Exploratory data analysis
  3. Data Preprocessing 
  4. Data Engineering
  5. Model Training
  6. Model Optimization
  7. Model Evaluation
  8. Conclusion from EDA and preprocessing
  9. Results
  10. Recommendations for future improvement
#
# Project Description:
Project is using Data Preprocessing, Exploratory Data Analysis, Regression modeling with a pipeline and hyperparameter optimization using Baseline models, TPOT, Random search and Grid search. Airline Delay dataset was taken from Kaggle: https://www.kaggle.com/datasets/giovamata/airlinedelaycauses
Since there were 5 delay columns I decided to combine them into one column by adding them up. This is the new target.

Features in Airline Delay dataset:

https://www.kaggle.com/datasets/giovamata/airlinedelaycauses/

1.	Year 2008
2.	Month 1-12
3.	DayofMonth 1-31
4.	DayOfWeek 1 (Monday) - 7 (Sunday)
5.	DepTime actual departure time (local, hhmm)
6.	CRSDepTime scheduled departure time (local, hhmm)
7.	ArrTime actual arrival time (local, hhmm)
8.	CRSArrTime scheduled arrival time (local, hhmm)
9.	UniqueCarrier unique carrier code
10.	FlightNum flight number
11.	TailNum plane tail number: aircraft registration, unique aircraft identifier
12.	ActualElapsedTime in minutes
13.	CRSElapsedTime in minutes
14.	AirTime in minutes
15.	ArrDelay arrival delay, in minutes: A flight is counted as "on time" if it operated less than 15 minutes later the scheduled time shown in the carriers' Computerized Reservations Systems (CRS).
16.	DepDelay departure delay, in minutes
17.	Origin origin IATA airport code
18.	Dest destination IATA airport code
19.	Distance in miles
20.	TaxiIn taxi in time, in minutes
21.	TaxiOut taxi out time in minutes
22.	Cancelled *was the flight cancelled
23.	CancellationCode reason for cancellation (A = carrier, B = weather, C = NAS, D = security)
24.	Diverted 1 = yes, 0 = no
25.	CarrierDelay in minutes: Carrier delay is within the control of the air carrier. Examples of occurrences that may determine carrier delay are: aircraft cleaning, aircraft damage, awaiting the arrival of connecting passengers or crew, baggage, bird strike, cargo loading, catering, computer, outage-carrier equipment, crew legality (pilot or attendant rest), damage by hazardous goods, engineering inspection, fueling, handling disabled passengers, late crew, lavatory servicing, maintenance, oversales, potable water servicing, removal of unruly passenger, slow boarding or seating, stowing carry-on baggage, weight and balance delays.
26.	WeatherDelay in minutes: Weather delay is caused by extreme or hazardous weather conditions that are forecasted or manifest themselves on point of departure, enroute, or on point of arrival.
27.	NASDelay in minutes: Delay that is within the control of the National Airspace System (NAS) may include: non-extreme weather conditions, airport operations, heavy traffic volume, air traffic control, etc.
28.	SecurityDelay in minutes: Security delay is caused by evacuation of a terminal or concourse, re-boarding of aircraft because of security breach, inoperative screening equipment and/or long lines in excess of 29 minutes at screening areas.
29.	LateAircraftDelay in minutes: Arrival delay at an airport due to the late arrival of the same aircraft at a previous airport. The ripple effect of an earlier delay at downstream airports is referred to as delay propagation.

#
---
# CONCLUSION from preliminary analysis of a dataset:
---
### Shape of a Dataset:     
Shape of the dataset is: 1936758 rows x 29 columns (without index column).

### Target Class
Target class is combination of all of the delay columns into one column by adding them up. This is the new target and after the NaN values in the individual columns are deleted, they will be summarized in one column.

### NaN values:  
There are 689270 rows with NaN values from the target features that should be deleted, because this is the target class we are predicting, imputation is not allowed.

There were other columns with NaN values, but after deleting the NaN from target features, other NaN values were also deleted, because they were in the same rows with the deleted ones.

There were only 2 NaN values in one of the columns, and they were deleted as well.

Deletion is done below in the code.

### Constant values:  
Column 'Year' has only one constant value so it will be deleted.

After rows deletion with NaN values, it was observed that three more columns had constant values: 'CancellationCode', 'Cancelled', 'Diverted'. They will be deleted from the dataset as well.  

### Non-informative and semantic irrelevant features
Columns 'FlightNum' and 'DayOfMonth' in my opinion are semantic irrelevant features and should be deleted from the dataset. 

### Data types:  
Except for date columns that should be concatanated into one datetime column, all other features have appropriate data type.

### Duplicates:  
There are 2 duplicate rows in the dataset and they will be deleted.

### Typos:       
There are no typos that need to be corrected.

### Descriptive statistics:
The summary statistics for the numerical columns in the dataset shows quick overview of the distribution and data variability that occurs in all columns. 
Column 'year' has constant value and should be deleted.
It is recommended to scale the features in order to ensure that they are on a similar scale, because some ML algorithms are sensitive to the scale of the features.

---
# CONCLUSION from EDA:

---
## Target class analysis: 
The target class is exponentially distributed and highly unbalanced, so therefore the best way is to use Stratified K-Folds cross-validaton in order to compute different test scores on different folds of the data. Since the target class has 1044 unique values, it needs to be digitized by converting them into a discrete variables.
The target class is time of delay in minutes, so therefore I decided that bins should be digitized within one hour range. All of the outliers are left in the last bin, because they carry meaningfull information in themselves, I decided not to delete them.

## Key findings from Heatmap:
##### Correlation Target - Features
**1)** Heatmap shows high positive correlation of 95-100% between target class: 'Delay' and features: 'ArrDelay' and 'DepDelay'. 
These features are the almost the same as target class, in fact they are not predictors, so they will be deleted.   
**2)** Target class has low correlation with all other features, but this is not enough reason to remove them from the model.  

##### Correlation Feature - Feature

**1)** Heatmap shows high positive correlation between the following features: 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'Distance' ('CRSElapsedTime', 'AirTime' and 'Distance' will be deleted because they have lower correlation with target class then 'ActualElapsedTime')  
**2)** There are other colums which are highly correlated 'DepTime, CRSDeptTime', 'AirTime', 'CSRArrTime', with correlation of 40-84%, but I decided to leave them in the dataset, and maybe to try to remove one or two of them later on, just for comparison with the first model. 

## Key findings from Histplots:  
Many of the columns have exponental distribution skewed to the left.

## Key findings from Boxplots:
Transformation was made with sqrt function on three columns, which resulted in reducing the skewness in these features. 

## Transformation on the features and target with skewed distribution:
**1)** **Features**: Transformations were made by applying np.sqrt() function on the skewed features.   
**2)** **Target**:   I decided to try transformed and not-transforemed version of the target class, to see how it will performs, because this it the most important column in the dataset.    

## Key findings from Pairplot:
Pairwise relationships between variables in a dataset here unlike the other graphs, has hue in bins from the Target Class. 
There is evident pattern in few of the plots, which looks similar to linear dependence. These features were highly correlated in the Heatmap: 'DepTime, CRSDeptTime', 'AirTime', 'CSRArrTime' so there is obvious correlation from this plot as well. 

Also can be noted that: **the most frequent delays of the planes are recorded from 15-60 min. Delays of up to 2 and 3 hours have much fewer occurrences, while the rest of the delays are very rare.**


## NOTE: 
The Conclusion was written after 2 versions were tested. Other versions are not documented in detaill, but they were discused on class.

---

#
---
# RESULTS:

---
## 1) Baseline models: 

#### Version I - Dataset WITHOUT Log transformation on target class
After evaluating multiple baseline models with default hyperparameters on our dataset, RandomForestRegressor() performed the best.  
DummyRegressor on the other hand, shows good stratification of the dataset splits, with R2 score: 0 %, which is very important because the dataset is highly unbalanced.

#### Version II - Dataset WITH Log transformation on target class
There is a slight improvement in all the models than in previous case.

#### Version III - Dataset WITHOUT Log transformation on target class and Additional Deletation on correlated colums
RandomForestRegressor() performed the best. This time score is very low compared to previous models.

#### Version IV - Dataset WITHOUT Log transformation on target class and Delay Substraction
This model has the lowest performance among all the models evaluated. 

---

## 2) Model Hyperparameter Optimization with Random Search: 

#### Version I - Dataset WITHOUT Log transformation on target class
RandomForestRegressor() once again performed the best. 

#### Version II - Dataset WITH Log transformation on target class
There is a slight improvement in Random Forest, but other models are way worst than in previous case.

#### Version III - Dataset WITHOUT Log transformation on target class and Additional Deletation on correlated colums
RandomForestRegressor() performed the best. KNN has also high score in this model.

#### Version IV - Dataset WITHOUT Log transformation on target class and Delay Substraction
Unlike Baseline model who has the lowest performance on this dataset, Random Forest performed excelent. Even KNN has high score. 

---

## 3) Model Hyperparameter Optimization with Grid Search: 

#### Version I - Dataset WITHOUT Log transformation on target class
The result is worse than all other models: Baseline models, Grid Search and TPOT optimization. I only managed to test GRID Search on only 1% of the dataset. I wasn't able to test it on whole dataset, not even on 10% of the dataset, because it took more than 8 hours for 10% od the dataset and it still wasn't finished so I terminated the program.
I think that maybe smaller result compared to other optimization models is consequence of the small dataset that was tested.

---

## 4) Model Hyperparameter Optimization with TPOT (Genetic optimization algorithms) : 
#### Version I - Dataset WITHOUT Log transformation on target class 
The result is worse than Baseline models, but I only managed to test TPOT on 10% of the dataset. I wasn't able to test it on whole dataset, because it took more than 12 hours and it still wasn't finished so I terminated the program.

---

#####

## RECOMMENDATIONS FOR FUTURE IMPROVEMENT:
---
In the future, we can focus on the following areas to further enhance our model's performance:

### * Slight changes in Features: 
Next step is to make some changes in the feature columns. For start we can try to remove two of the three columns that are highly correlated, with 71 - 84%.     
Try with removing the TailNumber column, because it has a lot of unique values and therefore its predictive power might be lower.    
Although I deleted the Distance column, I think it would be good to try and leave it, because it is a direct variable that indicates the exact distance between two destinations. Other variables such as flight duration and categorical variables such as Destination and Origin do not reflect the distance variable as exact number.

### * Feature Engineering: 
Some additional feature transformation to see how they will reflect the model performance. Maybe log transformation instead of sqrt transformation.

### * Hyperparameter Tuning: 
Using optimization algorithms for hyperparameter tuning, especially Genetic Algorithms, can lead to increased model performance.

### * Ensemble Methods: 
Implementing additional ensemble techniques may help capture complex relationships in the data.

By addressing all of these areas, we aim to further enhance the accuracy and reliability of our predictive model.

#

#
# License
MIT License
#

