#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df1=pd.read_csv("business.csv")
df2=pd.read_csv("economy.csv")


# In[4]:


data = pd.merge(df1, df2,how='outer')


# In[5]:


data.head()


# In[6]:


data['dep_time']


# In[7]:


data['dep_time'] = pd.to_datetime(data['dep_time'], format='%H:%M')


# In[8]:


data['departure_hour'] = data['dep_time'].dt.hour
data['departure_minute'] = data['dep_time'].dt.minute


# In[9]:


data[['departure_hour', 'departure_minute']] = data[['departure_hour', 'departure_minute']].apply(pd.to_numeric, errors='coerce')


# In[10]:


data.drop('dep_time', axis=1, inplace=True)


# In[11]:


data['arr_time'] = pd.to_datetime(data['arr_time'], format='%H:%M')


# In[12]:


data['arrived_hour'] = data['arr_time'].dt.hour
data['arrived_minute'] = data['arr_time'].dt.minute


# In[13]:


data[['arrived_hour', 'arrived_minute']] = data[['arrived_hour', 'arrived_minute']].apply(pd.to_numeric, errors='coerce')


# In[14]:


data.drop('arr_time', axis=1, inplace=True)


# In[15]:


data['time_taken']


# In[16]:


data[['timetaken_hours', 'timetaken_minutes']] = data['time_taken'].str.extract(r'(\d+)h (\d+)m')


# In[17]:


data[['timetaken_hours', 'timetaken_minutes']] = data[['timetaken_hours', 'timetaken_minutes']].apply(pd.to_numeric, errors='coerce')


# In[18]:


data.head()


# In[19]:


data.isnull().sum()


# In[20]:


data['timetaken_hours'].fillna(data['timetaken_hours'].mode()[0], inplace=True)
data['timetaken_minutes'].fillna(data['timetaken_minutes'].mode()[0], inplace=True)


# In[21]:


data['timetaken_hours'] = data['timetaken_hours'].round().astype(int)


# In[22]:


data['timetaken_minutes'] = data['timetaken_minutes'].round().astype(int)


# In[23]:


data.drop('time_taken', axis=1, inplace=True)


# In[24]:


data['date']


# In[25]:


data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year


# In[26]:


data.drop('date', axis=1, inplace=True)


# In[27]:


data[['day', 'month','year']] = data[['day', 'month','year']].apply(pd.to_numeric, errors='coerce')


# In[28]:


data['flight_code'] = data['ch_code'].astype(str) + data['num_code'].astype(str)


# In[29]:


data['stop']


# In[30]:


data['stop'] = data['stop'].str.replace('\n\t', '').str.strip()


# In[31]:


data['airline'] = data['airline'].str.lower()
data['from'] = data['from'].str.lower()
data['to'] = data['to'].str.lower()
data['flight_code'] = data['flight_code'].str.lower()



# In[32]:


data.drop('num_code', axis=1, inplace=True)
data.drop('ch_code',axis=1, inplace=True)


# In[33]:


data['price']


# In[34]:


data['price'] = pd.to_numeric(data['price'].str.replace(',', ''), errors='coerce')


# In[35]:


data['price'] = data['price'].astype(int)


# In[36]:


data['price'] = data['price'].round().astype(int)


# In[62]:


data.head(100)


# In[38]:


data['stop']


# In[39]:


data['stop'] = data['stop'].str.replace('-', '')


# In[40]:


data.describe()


# statistical analysis

# In[41]:


correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[42]:


##checking outlier
# Example: Boxplot of departure delays
sns.boxplot(x='price', data=data)
plt.title('Boxplot of price')
plt.show()


# In[43]:


#remove outliers
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)

# Calculate IQR (Interquartile Range)
IQR = Q3 - Q1

# Define lower and upper bounds to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers based on the bounds
outliers = ((data['price'] < lower_bound) | (data['price'] > upper_bound))

# Replace outliers with NaN (or any other value)
data['price'][outliers] = np.nan


# In[44]:


data['price'].fillna(data['price'].mode()[0], inplace=True)


# In[45]:


from mpl_toolkits.mplot3d import Axes3D

# Assuming 'df' is your DataFrame
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for DepartureTime, TimeTaken, and Price
ax.scatter(data['departure_hour'], data['timetaken_hours'], data['price'], c='blue', marker='o')

ax.set_xlabel('departure_hour')
ax.set_ylabel('timetaken_hours')
ax.set_zlabel('price')

plt.title('3D Scatter Plot of Departure Time, Time Taken, and Price')
plt.show()


# In[46]:


sns.scatterplot(x='departure_hour', y='timetaken_hours', hue='airline', data=data, palette='viridis')
plt.title('Scatter Plot of Departure Time vs. Time Taken (Colored by Airline)')
plt.xlabel('Departure Time')
plt.ylabel('Time Taken')
plt.legend(title='Airline', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[51]:


data['month'] = data['month'].astype(int)
data['year'] = data['year'].astype(int)
data['day'] = data['day'].astype(int)


# In[52]:


monthly_flight_count = data.groupby(['year', 'month']).size().reset_index(name='flight_count')


# In[55]:


plt.figure(figsize=(12, 8))
sns.pointplot(x='month', y='flight_count', hue='year', data=monthly_flight_count, dodge=True, markers='o')

# Customize the plot
plt.title('Monthly Flight Trends')
plt.xlabel('Month')
plt.ylabel('Flight Count')

# Show the plot
plt.show()


# In[57]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='stop', y='price', data=data)

# Customize the plot
plt.title('Price Variation by Stops')
plt.xlabel('Number of Stops')
plt.ylabel('Price')

# Show the plot
plt.show()


# In[58]:


avg_price_by_route = data.groupby(['from', 'to'])['price'].mean().reset_index()


# In[59]:


heatmap_data = avg_price_by_route.pivot('from', 'to', 'price')
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f", linewidths=.5)

# Customize the plot
plt.title('Average Price by Route')
plt.xlabel('from')
plt.ylabel('to')


# In[61]:


city_destination_counts = data.groupby(['from', 'to']).size().reset_index(name='count')

# Find the top 10 starting cities
top_start_cities = city_destination_counts.groupby('from')['count'].sum().sort_values(ascending=False).head(10).index

# Filter the data for the top 10 starting cities
top_cities_data = city_destination_counts[city_destination_counts['from'].isin(top_start_cities)]

# Bar plot to visualize the top 10 starting cities with their most common destinations
plt.figure(figsize=(14, 8))
sns.barplot(x='from', y='count', hue='to', data=top_cities_data)

# Customize the plot
plt.title('Top 10 Starting Cities with Most Common Destinations')
plt.xlabel('from')
plt.ylabel('Count')
plt.legend(title='Destination', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()


# In[63]:


plt.figure(figsize=(14, 8))
sns.boxplot(x='airline', y='price', data=data)

# Customize the plot
plt.title('Price Distribution by Airline')
plt.xlabel('Airline')
plt.ylabel('Price')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


# In[64]:


avg_price_by_flight_year = data.groupby(['flight_code', 'year'])['price'].mean().reset_index()

# Find the top 10 flights based on the highest average prices
top_flights = avg_price_by_flight_year.sort_values(by='price', ascending=False).head(10)

# Pie chart to visualize the distribution of average prices among the top flights
plt.figure(figsize=(10, 8))
plt.pie(top_flights['price'], labels=top_flights['flight_code'], autopct='%1.1f%%', startangle=140)

# Customize the plot
plt.title('Distribution of Average Prices Among Top 10 Flights')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Show the plot
plt.show()


# In[69]:


data.head()


# In[70]:


##avg time taken by airline
avg_time_taken_by_airline = data.groupby('airline')['timetaken_hours'].mean().reset_index()

# Bar plot to visualize the average time taken for each airline
plt.figure(figsize=(12, 8))
plt.bar(avg_time_taken_by_airline['airline'], avg_time_taken_by_airline['timetaken_hours'], color='skyblue')

# Customize the plot
plt.title('Average Time Taken by Airline')
plt.xlabel('Airline')
plt.ylabel('Average Time Taken')
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


# In[71]:


print(data.dtypes)


# In[72]:


data.to_csv('flight_data.csv', index=False)


# In[ ]:




