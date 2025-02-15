# Australian Vehicle Price Prediction Using PySpark

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
  - [Linear Regression Model](#linear-regression-model)
- [Model Evaluation](#model-evaluation)
- [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction
This project aims to predict vehicle prices in the Australian market using PySpark. The dataset includes multiple features like car type, fuel type, engine type, and more. The goal is to perform data preprocessing, exploratory data analysis (EDA), and build a linear regression model to predict vehicle prices based on these features.

## Dataset Overview
The dataset used in this analysis contains various vehicle attributes, including:
- **Make**: The brand of the vehicle.
- **Model**: The model of the vehicle.
- **Year**: The year of manufacture.
- **Kilometres**: The total kilometres driven.
- **Fuel Type**: Type of fuel used (e.g., Petrol, Diesel, Electric).
- **Transmission**: The type of transmission used (e.g., Manual, Automatic).
- **Body Type**: Type of the vehicle (e.g., Sedan, SUV).
- **Price**: The target variable representing the vehicle price.

The dataset provides a rich set of features to explore the factors that influence vehicle pricing in the Australian market.

## Data Preprocessing
The preprocessing steps include:
- **Missing Data**: We handled missing values in important columns by dropping rows with missing target values (Price) and imputing others where necessary.
- **Data Type Conversion**: We ensured that numerical columns were properly cast to the correct data types (e.g., `Year`, `Kilometres`, `Price`).
- **Categorical Encoding**: Categorical variables such as 'FuelType', 'Transmission', and 'BodyType' were encoded using PySpark's string indexer and one-hot encoding.
- **Feature Scaling**: The numeric features were scaled for better performance of the machine learning model, particularly when using linear regression.

## Exploratory Data Analysis (EDA)
The EDA was conducted to uncover relationships and patterns within the dataset:
- **Price Distribution**: We visualized the distribution of vehicle prices, noting the skewness and outliers in the data.
- **Top Car Brands**: The most popular car brands in the dataset were identified.
- **Fuel Type vs. Price**: A comparison of different fuel types and their impact on vehicle prices.
- **Transmission vs. Price**: We analyzed the effect of transmission type on price.
- **Body Type Analysis**: The distribution of vehicle types (e.g., Sedan, SUV) and their corresponding prices.

## Modeling

### Linear Regression Model
The linear regression model was built using PySpark’s `LinearRegression` class to predict the vehicle prices. The features were selected based on the results of the EDA and included variables such as `Year`, `Kilometres`, `FuelType`, `Transmission`, and `BodyType`. The model was trained on a training dataset and tested on a validation set.

```python
lr = LinearRegression(featuresCol='scaled_features', labelCol='Price')
lr_model = lr.fit(train_lr_data)
```

## Model Evaluation
The performance of the linear regression model was evaluated using the following metrics:
- **Root Mean Squared Error (RMSE)**: 22,579.5 - Measures the average deviation between predicted and actual values.
- **Mean Absolute Error (MAE)**: 12,217.5 - Measures the average magnitude of the errors.
- **R-Squared (R²)**: 0.538 - Indicates that the model explains 53.8% of the variance in the target variable (Price).

```python
rmse = evaluator.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)
```

## Results and Discussion

### Model Performance
The linear regression model performed adequately but was not perfect. With an R-squared value of **0.538**, it explains only 53.8% of the variance in car prices, which suggests there are other factors not accounted for in the model. This moderate performance could be improved with additional features, more sophisticated models, or non-linear relationships.

### Limitations
- **Skewed Price Distribution**: Vehicle prices in the dataset show a right skew, meaning the model may struggle with predicting high-priced vehicles accurately.
- **Linear Relationships**: The linear regression model assumes linear relationships between the features and the target, which might not fully capture the complexity of the pricing mechanisms in the dataset.

### Future Work
- **Advanced Models**: Experimenting with models like Random Forest or Gradient Boosting could improve prediction accuracy by capturing non-linear relationships in the data.
- **Feature Engineering**: Adding more features, such as geographic location or additional vehicle specifications, could further improve the model's performance.

## Conclusion
The linear regression model provides a baseline for predicting Australian vehicle prices. While it gives valuable insights, more complex models and additional features would likely enhance prediction accuracy. This analysis can help understand key factors influencing vehicle pricing, benefiting businesses in the automotive industry.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
