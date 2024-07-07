# Data Analysis and Cleaning Project

## Overview
This project focuses on data analysis and cleaning using Python. It involves working with datasets related to global air pollution and world cities, demonstrating various data cleaning, analysis, and visualization techniques.

## Features
- Data cleaning and preprocessing.
- Exploratory Data Analysis (EDA).
- Data visualization using various Python libraries.
- Merging datasets and handling missing values.
- Performing clustering and linear regression analysis.

## Technologies and Tools Used
- **Python:** The primary programming language used.
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical computations.
- **Matplotlib:** For creating static visualizations.
- **Seaborn:** For statistical data visualization.
- **Plotly:** For interactive visualizations.
- **scikit-learn:** For machine learning algorithms.
- **math:** For mathematical operations.
- **random:** For generating random numbers.

## Getting Started
1. Clone the repository.
   ```sh
   git clone https://github.com/DanielB66/Data-analysis-and-cleaning-project
   ```
2. Install the required libraries.
   ```sh
   pip install -r requirements.txt
   ```
3. Run the `main.py` script to perform the data analysis and visualizations.

## Datasets
- **updated_global_air_pollution_dataset.csv:** Contains data on air pollution from various cities around the world.
- **worldcities.csv:** Contains data on cities around the world, including latitude, longitude, population, and other attributes.

## Analysis Steps
1. **Data Loading:**
   - Load the datasets using Pandas.
   - Display basic information about the datasets.

2. **Data Cleaning:**
   - Handle missing values.
   - Convert data types as necessary.
   - Remove duplicates and handle invalid values.

3. **Data Merging:**
   - Perform a full outer join on the datasets based on the city names.

4. **Exploratory Data Analysis (EDA):**
   - Display summary statistics and data distributions.
   - Visualize data using various plots (bar plots, heatmaps, violin plots, scatter plots, etc.).

5. **Normalization and Scaling:**
   - Normalize specific columns to prepare for analysis.

6. **Clustering:**
   - Use the K-Means algorithm to cluster cities based on population and AQI values.

7. **Linear Regression:**
   - Perform linear regression to analyze the relationship between population and AQI values.

## Visualizations
- **Bar Plots:** For comparing average AQI values per country.
- **Heatmaps:** For showing correlation matrices.
- **Violin Plots:** For visualizing the distribution of AQI values by country.
- **Scatter Plots:** For analyzing relationships between population and AQI values.
- **Histograms:** For visualizing the distribution of AQI values.
- **Line Plots:** For showing trends in AQI values over time for selected cities.
- **Pair Plots:** For exploring relationships between multiple variables.
- **Pie Charts:** For showing population distribution by country.
- **Elbow Method Plots:** For determining the optimal number of clusters.

## Contributing
Feel free to fork this repository and contribute by submitting a pull request.

## License
This project is licensed under the MIT License.
