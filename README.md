# Housing Price Prediction Model
## Description
The purpose of the project is to attempt to accurately predict housing prices in California using the dataset provided by Kaggle. Some manipulation and clean-up of the data is required in order to extrapolate the best possible score for our model. Utilizing Machine Learning concepts such as a Convolutional Neural Network and the Random Forest Regression Model, we’re able to train a model that will more accurately predict future prices with Linear Regression.

## Dataset
The dataset contains the following columns:
>[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, ocean_proximity].

The dataset contained Null values in 207/20640 entries, as a result those entries were culled. 
For the train_test_split, we dropped median_house_value column for the X matrix, and used the same column as our y parameter.
> X = data.drop(['median_house_value'], axis=1) #Median house value is the target variable

> y = data['median_house_value']

> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #20 percent for testing

While the dataset on its own could be used as is, we will get better results if we do some data cleaning and manipulation to try to extract as much value from the information provided.
As it stands, if we plot the current information into histograms, we can see that we’re not able to draw good predictions due to the inconsistency between all the graphs.
<img src="/images/unmodified_histogram.png">

### Ocean Proximity
One potential variable that could provide value to our training would be the ocean_proximity column. 
However, it cannot be used as is due to the column containing non-numerical values. As such, I've decided to transpose the ocean_proximity column. The transposed column then returns the following:
> [<1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN]

Due to these columns providing True and False values that can be interpreted as 1 or 0. We can better determine if it makes a difference in our model.
Plotting the data along a Latitude vs Longitude graph we can see that proximity to an ocean does indeed correlate with higher house values:
<img src="/images/lat_long_ocean_proximity.png">
<img src="/images/ocean_proximity_heatmap.png">

### Rooms and Population
Within the dataset, we have the following columns 
> [total_rooms, total_bedrooms, population, households]

While on their own, the variables do have some impact on the housing value, they aren’t nearly as accurate without further manipulation. We can use the following formula to give us two new columns.
> train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']

> train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

The two ratios give us values between 0-1, which can provide more detailed info to the model as to how the values correlate.
When plotted on a heatmap, we see that these new variables provide a greater contribution to our existing data and thus would help the model predict more accurately.
<img src="/images/bedroom_ratio_household_rooms_heatmap.png">

# Training the Model
If we scale the model, and fit it directly into our *LinearRegression* function, we receive a value of 0.65.

While it’s not terrible, it’s not a good score. Improvements can be made by using Machine Learning to give us more accurate results. 
Using *RandomForestRegressor* from sklearn, we can try to increase the score and give us better predictions. 
When generating the forest, the initial parameters provided were 
> n_estimators = [10, 30, 100]

> min_samples_split = [2,4,8]

Using a grid search function we are able to try to acquire the best estimator for our model. 

After running it, we get the final values of 
> n_estimators = 100

> min_samples_split = 8


When we then score our dataset using these values as our parameters, we receive a value of 0.82.
Meaning our model through machine learning training became approximately 20% more accurate for determining median_house_values.


