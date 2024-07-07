import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib


# קריאת קבצי הנתונים
input1_df = pd.read_csv('updated_global_air_pollution_dataset.csv')
input2_df = pd.read_csv('worldcities.csv')

#  הדפסת מידע על מערך הנתונים
print("\ninfo for global air pollution.csv:\n")
print(input1_df.info())
print("\nColumns in global air pollution.csv:\n")
print(input1_df.columns)
print("\nData Types:\n")
print(input1_df.dtypes)
print("\nMissing Values:\n")
print(input1_df.isnull().sum())

print("\ninfo for worldcities.csv:\n")
print(input1_df.info)
print("\nColumns in worldcities.csv:\n")
print(input2_df.columns)
print("\nData Types:\n")
print(input2_df.dtypes)
print("\nMissing Values:\n")
print(input2_df.isnull().sum())

# הדפסת שורות ראשונות, אחרונות, ובאמצע
print('\nHead of global air pollution.csv:\n')
print(input1_df.head())

print('\nTail of global air pollution.csv:\n')
print(input1_df.tail(5))

print('\nSample of global air pollution.csv:\n')
print(input1_df.sample(5))

print('\ndescribe of global air pollution.csv:\n')
print(input1_df.describe(include='all'))

print('\nHead of worldcities.csv:\n')
print(input2_df.head())

print('\nTail of worldcities.csv:\n')
print(input2_df.tail(5))

print('\nSample of worldcities.csv:\n')
print(input2_df.sample(5))

print('\ndescribe of worldcities.csv:\n')
print(input2_df.describe(include='all'))

missing_percentage = (input1_df.isnull().sum() / len(input1_df)) * 100
print('\n% of Null values global_air_pollution_dataset:\n')
print(missing_percentage)

missing_percentage = (input2_df.isnull().sum() / len(input2_df)) * 100
print('\n% of Null values worldcities:\n')
print(missing_percentage)

input1_df['City'] = input1_df['City'].astype(str)
input2_df['City'] = input2_df['City'].astype(str)

# יצירת data-frame חדש ע"י פעולת Full Outer Join
outer_join_df = pd.merge(input1_df, input2_df, on='City', how='outer')

# בדיקת הערכים החסרים בקובץ הממוזג
print("\nMissing Values in Outer Join DataFrame:\n")
print(outer_join_df.isnull().sum())

missing_percentage = (outer_join_df.isnull().sum() / len(outer_join_df)) * 100
print('\n% of missing values outer_join_df:\n')
print(missing_percentage)


# בחירת עמודות שונות ויצירת שני data-frames חדשים
df1 = outer_join_df[['City', 'Country_x','iso3', 'population', 'AQI Value', 'PM2.5 AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'CO AQI Value']].copy()
df2 = outer_join_df[['City', 'Country_y', 'Lat', 'Lng', 'AQI Value', 'PM2.5 AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'CO AQI Value']].copy()

# פונקציה לבדיקת ערכים לא מתאימים בעמודות מספריות
def char_fixer(data_frame, series_name):
    cnt = 0
    for row in data_frame[series_name]:
        try:
            float(row)
        except ValueError:
            if str(row).lower() == 'true' or str(row).lower() == 'false':
                data_frame.drop([cnt], inplace=True)
            elif str(row).lower() == 'nan':
                data_frame.loc[cnt, series_name] = np.nan
            else:
                data_frame.drop([cnt], inplace=True)
        cnt += 1
    data_frame[series_name] = data_frame[series_name].astype('float64', errors='raise')
    data_frame.reset_index(drop=True, inplace=True)

# פונקציה לבדיקת ערכים לא מתאימים בעמודות טקסטואליות
def num_fixer(data_frame, series_name):
    cnt = 0
    for row in data_frame[series_name]:
        try:
            int(float(row))
        except ValueError:
            if str(row).lower() == 'true' or str(row).lower() == 'false':
                data_frame.drop([cnt], inplace=True)
            elif str(row).lower() == 'nan':
                data_frame.loc[cnt, series_name] = np.nan
        else:
                data_frame.drop([cnt], inplace=True)
        cnt += 1
    data_frame[series_name] = data_frame[series_name].astype('string', errors='raise')
    data_frame.reset_index(drop=True, inplace=True)

 #פונקציה להחלפת ערכים חסרים
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            mean_value = df[column].mean()
            df[column] = df[column].fillna(mean_value)
            print(f"Filled missing values in {column} with mean value: {mean_value}")
        elif column in ['City', 'Country_x', 'Country_y','iso3']:
            df[column] = df[column].fillna('Unknown')
            print(f"Filled missing values in {column} with default value: Unknown")
        elif df[column].dtype == 'object':
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)
            print(f"Filled missing values in {column} with mode value: {mode_value}")
    return df

# ניקוי עמודות מספריות עבור df1
for col in ['population','AQI Value', 'PM2.5 AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'CO AQI Value']:
    char_fixer(df1, col)

# ניקוי עמודות טקסטואליות עבור df1
for col in ['City', 'Country_x','iso3']:
    num_fixer(df1, col)

# ניקוי עמודות מספריות עבור df2
for col in [ 'Lat', 'Lng', 'AQI Value', 'PM2.5 AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'CO AQI Value']:
    char_fixer(df2, col)

# ניקוי עמודות טקסטואליות עבור df2
for col in ['City', 'Country_y']:
    num_fixer(df2, col)

# החלפת ערכים חסרים ב-df1 וב-df2
df1 = fill_missing_values(df1)
df2 = fill_missing_values(df2)

# הדפסת מידע על ה-dataframes הנקיים
print('\nClean DataFrame df1:')
print(df1.info())

print('\nClean DataFrame df2:')
print(df2.info())

# הדפסת שורות ראשונות מה-DataFrame הנקי
print('\nHead of Clean DataFrame df1:')
print(df1.head())

print('\nHead of Clean DataFrame df2:')
print(df2.head())


def normalize_column(df, column_name):
    max_value = df[column_name].abs().max()
    normalized_column_name = column_name + '_norm'
    df[normalized_column_name] = df[column_name].abs() / max_value
    print(f"Normalized {column_name} to {normalized_column_name}")

normalize_column(df1, 'population')

print('\nClean DataFrame input1_df after normalization:')
print(df1.info())

duplicate_rows = df1[df1.duplicated()]
print("\nDuplicate rows found in df1:")
print(duplicate_rows)

# הסרת שורות כפולות
print("\nRemoving duplicate rows from df1:")
df1.drop_duplicates(inplace=True)
print(df1.tail(15))

duplicate_rows1 = df1[df1.duplicated()]
print("\nDuplicate rows in :")
print(duplicate_rows1)

df1.to_excel('df1.xlsx', index=True)

matplotlib.use('Qt5Agg')
file_path = 'df1.xlsx'
df1 = pd.read_excel(file_path)
numeric_columns = ['population', 'AQI Value', 'PM2.5 AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'CO AQI Value']
df1[numeric_columns] = df1[numeric_columns].apply(pd.to_numeric, errors='coerce')
df1['AQI Value'] = df1['AQI Value'].fillna(df1['AQI Value'].mean())

# גרף עמודות של ממוצע ערכי AQI לפי מדינה
avg_aqi_per_country = df1.groupby('Country_x')['AQI Value'].mean().reset_index()
fig = px.bar(avg_aqi_per_country, x='Country_x', y='AQI Value', title='Average AQI Value per Country')
fig.update_layout(xaxis_title='Country', yaxis_title='Average AQI Value')
fig.show()

# Heatmap של מטריצת קורלציה
corr = df1[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Violin Plot של ערכי AQI לפי מדינה
plt.figure(figsize=(12, 8))
plt.violinplot([df1[df1['Country_x'] == country]['AQI Value'] for country in df1['Country_x'].unique()],
               showmeans=True)
plt.xticks(range(1, len(df1['Country_x'].unique()) + 1), df1['Country_x'].unique(), rotation=90)
plt.title('Violin Plot of AQI Values by Country')
plt.xlabel('Country')
plt.ylabel('AQI Value')
plt.show()

# יצירת תרשים פיזור צבעוני של גודל אוכלוסייה בערים בעולם מול ערכי זיהום אוויר AQI
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df1['population'], df1['AQI Value'], c=df1['AQI Value'], cmap='viridis')
plt.colorbar(scatter, label='AQI Value')
plt.xlabel('Population')
plt.ylabel('AQI Value')
plt.title('Colorful Scatter Plot of Population vs AQI Value')
plt.grid(True)
plt.show()


# יצירת גרף בר של ערכי AQI לפי 20 הערים הגדולות ביותר
city_population = df1.groupby('City')['population'].sum().reset_index()
city_population_filtered = city_population[city_population['City'] != 'Unknown']
top_cities_filtered = city_population_filtered.nlargest(20, 'population')
top_cities_data = df1[df1['City'].isin(top_cities_filtered['City'])]
plt.figure(figsize=(14, 8))
sns.barplot(data=top_cities_data, x='AQI Value', y='City',hue='City', palette='viridis', order=top_cities_filtered['City'])
plt.title('AQI Values for the Top 20 Cities by Population Size')
plt.xlabel('AQI Value')
plt.ylabel('City')
plt.show()


# יצירת גרף בר של גודל האוכלוסייה לפי 20 הערים הגדולות ביותר
city_population = df1.groupby('City')['population'].sum().reset_index()
city_population_filtered = city_population[city_population['City'] != 'Unknown']
top_cities_filtered = city_population_filtered.nlargest(20, 'population')
plt.figure(figsize=(14, 8))
sns.barplot(data=top_cities_filtered, x='population', y='City',hue='City', palette='viridis')
plt.title('Top 20 Cities by Population Size (Excluding Unknown)')
plt.xlabel('Population')
plt.ylabel('City')
plt.show()

# חישוב ממוצע זיהום האוויר לכל מדינה
avg_aqi_per_country = df1.groupby('Country_x')['AQI Value'].mean().reset_index()
# יצירת גרף ברים של זיהום האוויר לפי מדינות
fig = px.bar(avg_aqi_per_country, x='Country_x', y='AQI Value', title='Average AQI Value per Country')
fig.update_layout(xaxis_title='Country', yaxis_title='Average AQI Value')
fig.show()

# גרף עוגה (Pie Chart)
country_population = df1.groupby('Country_x')['population'].sum().reset_index()
country_population = country_population[country_population['Country_x'] != 'Unknown']
fig = px.pie(country_population, values='population', names='Country_x', title='Population Distribution by Country')
fig.show()


# היסטוגרמה (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(df1['AQI Value'], bins=30, kde=True)
plt.title('Distribution of AQI Values')
plt.xlabel('AQI Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# יצירת גרף קווי מרובה סדרות
# נבחר מספר ערים לדוגמה
cities = ['Paris', 'New York', 'Tokyo', 'Mumbai']
plt.figure(figsize=(14, 8))
for city in cities:
    city_data = df1[df1['City'] == city]
    plt.plot(city_data.index, city_data['AQI Value'], marker='o', label=city)
plt.xlabel('Index')
plt.ylabel('AQI Value')
plt.title('AQI Value Trends for Selected Cities')
plt.legend()
plt.grid(True)
plt.show()

# יצירת גרף זוגות (Pairplot)
sns.pairplot(df1[['population', 'AQI Value', 'PM2.5 AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'CO AQI Value']])
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

# בחירת העמודות המתאימות לחלוקה לאשכולות
X = df1[['population', 'AQI Value']]

# נרמול הנתונים
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# מבחן המרפק לקביעת מספר האשכולות הנכון
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances (SSE)')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

# חלוקה לאשכולות לפי מספר האשכולות הנכון
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df1['Cluster'] = kmeans.fit_predict(X_scaled)

# הצגת האשכולות בגרף
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df1['population'], df1['AQI Value'], c=df1['Cluster'], cmap='viridis', marker='o')
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Population')
plt.ylabel('AQI Value')
plt.title('Clusters of Population and AQI Value')
plt.grid(True)
plt.show()

# הצגת מספר האשכולות ומסקנות
print(f"Number of clusters chosen: {optimal_k}")
print("Conclusion: The data has been divided into clusters, each representing cities with similar population and AQI values. The clustering helps identify patterns in the data.")

# בחירת שתי עמודות לרגרסיה ליניארית
X = df1[['population']]
y = df1['AQI Value']

# יצירת מודל הרגרסיה
reg = LinearRegression()
reg.fit(X, y)

# חישוב קו הרגרסיה
y_pred = reg.predict(X)

# תרשים פיזור עם קו הרגרסיה
plt.figure(figsize=(10, 6))
sns.scatterplot(x='population', y='AQI Value', data=df1, color='blue', alpha=0.5)
plt.plot(df1['population'], y_pred, color='red')
plt.xlabel('Population')
plt.ylabel('AQI Value')
plt.title('Linear Regression: Population vs AQI Value')
plt.grid(True)
plt.show()

# הצגת מקדם הרגרסיה והערך החותך
print(f"Regression Coefficient (Slope): {reg.coef_[0]}")
print(f"Intercept: {reg.intercept_}")

# מסקנות הרגרסיה
print("Conclusion: The linear regression model shows the relationship between the population and AQI Value. The red line represents the best fit line based on the linear regression. The slope of the line indicates the average change in AQI Value for a unit change in population.")

