# Housing Price Prediction Model
## Purpose
The purpose of the project is to attempt to accurately predict housing prices in California using the dataset provided. Utilizing Machine Learning concepts such as a Convolutional Neural Network and  the Random Forest Regression Model, we’re able to more accurately predict future prices for housing based on a number of values provided.

## Specifics
### Dataset
The dataset contains the following columns:
[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, ocean_proximity]
The dataset contained Null values in 207/20640 entries, as a result those entries were culled. 
For the train_test_split, we dropped median_house_value column for the X matrix, and used the same column as our y parameter.
X = data.drop(['median_house_value'], axis=1) #Median house value is the target variable
y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #20 percent for testing
While the dataset on its own could be used as is, we will get better results if we do some data cleaning and manipulation to try to extract as much value from the information provided.
As it stands, if we plot the current information into histograms, we can see that we’re not able to draw good predictions due to the inconsistency between all the graphs.
*INSERT IMAGE HERE*
#### Ocean Proximity
A suspected variable that could provide a great deal of value to our project would be the ocean_proximity column. However, it cannot be used as is as the variables are strings and aren’t providing numerical values. My solution is to transpose the ocean_proximity column which returns the values provided as columns with True/False values. The following columns are provided after transposing [<1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN]
If we then plot this information along a Latitude vs Longitude graph we can see that proximity to an ocean does indeed correlate with higher median_house_value:
*INSERT IMAGE HERE*

#### Rooms and Population
The columns [total_rooms, total_bedrooms, population, households], while on their own do have some impact on the housing value, aren’t nearly as accurate without further manipulation. We can use the following formula to condense  the columns into [bedroom_ratio, household_rooms]:
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']
When plotted on a heatmap, we see that these new variables provide a greater correlation to median_house_value than having those values on their own.
*INSERT IMAGE HERE*
## Training the Model
If we scale the model, and fit it directly into our LinearRegression function, we receive a value of 0.65.
While it’s not terrible, it’s not a good score. Improvements can be made by using Machine Learning to give us more accurate results. Using RandomForestRegressor from sklearn, we can try to increase the score so that predictions are more accurate. 
When generating the forest, the initial parameters provided were n_estimators = [10, 30, 100] and min_samples_split = [2,4,8]. Using a grid search function we are able to try to acquire the best estimator for our model. 
After running it, we get the final values of n_estimators = 100 and min_samples_split = 8 for the optimal variables for our forest.
When we then score our test dataset, we receive a value of 0.82.
Meaning our model through machine learning training became approximately 20% more accurate for determining median_house_values.


