# Lab 4 Report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

%matplotlib inline

from IPython.display import Image as ipyimage #For displaying images in colab jupyter cell

### Exercise 1: Construct Dictionaries from Data

ipyimage('lab4_exercise1.PNG', width = 1000)

def convert_csv_to_dict(file_path):
    
    df = pd.read_csv(file_path)
    
    # Extract the filename using delimiters
    filename = file_path.split('/')[-1]
    
    # Initialise dictionary object with filename
    dictionary_obj = {'Filename': filename}
    
    # Convert columns through numpy
    for column in df.columns:
        dictionary_obj[column] = df[column].to_numpy()
    
    return dictionary_obj

TSLA_dict = convert_csv_to_dict("C:/Users/ankit/Downloads/Lab4_Report_Template/Lab4_Report_Template/TSLA.csv")

# Navigate to keys corresponding to 2nd and 4th columns (Open and Low prices) of TSLA.csv, 
# Print first 10 elements of each key.

TSLA_column2 = "Open"
TSLA_column4 = "Low"
print(TSLA_dict[TSLA_column2][:10])
print(TSLA_dict[TSLA_column4][:10])

diabetes_dict = convert_csv_to_dict("C:/Users/ankit/Downloads/Lab4_Report_Template/Lab4_Report_Template/diabetes.csv")

# Navigate to keys corresponding to 2nd and 4th columns (Glucose and SkinThickness) of diabetes.csv, 
# Print first 10 elements of each key.

diabetes_column2 = "Glucose"
diabetes_column4 = "SkinThickness"
print(diabetes_dict[diabetes_column2][:10])
print(diabetes_dict[diabetes_column4][:10])

### Exercise 2: Bar graph with confidence intervals

ipyimage('lab4_exercise2.PNG', width = 1000)

# Load diabetes.csv 
# Split the data into diabetic and non-diabetic

diabetes = pd.read_csv('diabetes.csv')

non_diabetic = diabetes[diabetes['Outcome'] == 0]
diabetic = diabetes[diabetes['Outcome'] == 1]

# Extract glucose, blood pressure, and BMI metrics from diabetic and non-diabetic

# non-diabetic metrics
non_diabetic_glucose = non_diabetic['Glucose'].dropna().values
non_diabetic_bp = non_diabetic['BloodPressure'].dropna().values
non_diabetic_bmi = non_diabetic['BMI'].dropna().values

# diabetic metrics
diabetic_glucose = diabetic['Glucose'].dropna().values
diabetic_bp = diabetic['BloodPressure'].dropna().values
diabetic_bmi = diabetic['BMI'].dropna().values

non_diabetic_list = [non_diabetic_glucose, non_diabetic_bp, non_diabetic_bmi]
diabetic_list = [diabetic_glucose, diabetic_bp, diabetic_bmi]

non_diabetic_bar_labels = ['Non-diabetic Glucose', 'Non-diabetic BP', 'Non-diabetic BMI']
diabetic_bar_labels = ['Diabetic Glucose', 'Diabetic BP', 'Diabetic BMI']

def produce_bargraph_CI(data_vec_list, conf_level, bar_labels):
    
    # Calculate means and confidence intervals
    means = [np.mean(data) for data in data_vec_list]
    n = [len(data) for data in data_vec_list]
    std_err = [stats.sem(data) for data in data_vec_list]
    h = [std_err[i] * stats.t.ppf((1 + conf_level) / 2., n[i]-1) for i in range(len(std_err))]

    # Plotting the bar graph
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(means)), means, yerr=h, alpha=0.7, color='blue', capsize=10)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(bar_labels, rotation=45, ha='right')
    ax.set_title('Confidence intervals'.format(conf_level), fontsize=15)
    ax.set_ylabel('Metric Values', fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval,2), ha='center', va='bottom', fontsize=10)

    # Show the bar graph
    plt.tight_layout()
    plt.show()


produce_bargraph_CI(data_vec_list = non_diabetic_list, conf_level = 0.99, bar_labels = non_diabetic_bar_labels)

produce_bargraph_CI(data_vec_list = diabetic_list, conf_level = 0.95, bar_labels = diabetic_bar_labels)

### Exercise 3: Rolling Mean/Median Function from Scratch

ipyimage('lab4_exercise3.PNG', width = 1000)

# Load stock datasets

tesla = pd.read_csv('TSLA.csv') 
tesla_np = tesla.to_numpy()

google = pd.read_csv('GOOGL.csv') 
google_np = google.to_numpy()

dji = pd.read_csv('DJI.csv') 
dji_np = dji.to_numpy()

# Extract closing price for each stock data

tesla_np_closing = tesla_np[:, 4]
google_np_closing = google_np[:, 4]
DJI_np_closing = dji_np[:, 4]

def smooth_data(data_arr, smooth_type, window_size):
    
    # Calculate the padding width
    pad_width = window_size // 2
    
    # Pad the original array with zeros only at the beginning
    padded_arr = np.pad(data_arr, (pad_width, 0), mode='constant', constant_values=0)
    
    # Initialize the smoothed array with zeros
    smoothed_data_arr = np.zeros(data_arr.shape)
    
    # Apply the smoothing
    for i in range(len(data_arr)):
        # Adjust the index to start from the padded part
        window_elements = padded_arr[i:i+window_size]
        if smooth_type == 'mean':
            smoothed_data_arr[i] = np.mean(window_elements)
        elif smooth_type == 'median':
            smoothed_data_arr[i] = np.median(window_elements)
        else:
            print("smooth_type must be 'mean' or 'median'")
            return None  # Return None if smooth_type is incorrect 
    
    return smoothed_data_arr



# Tesla closing prices, smooth_type = 'mean', window_size = 100
# Your smoothed data should be the same dimension as the original

smoothed_tsla_closing  = smooth_data(tesla_np_closing, smooth_type = 'mean', window_size = 100)

# plot smoothed_tsla_closing on top of tesla_np_closing

# YOUR CODE HERE
plt.figure(figsize=(14, 7))

# Original Tesla closing prices
plt.plot(tesla_np_closing, label='Original Tesla Closing Prices', linestyle='-')

# Smoothed Tesla closing prices
plt.plot(smoothed_tsla_closing, label='Smoothed Tesla Closing Prices (Mean, Window=100)', linestyle=':')

plt.title('Tesla Closing Prices and Smoothed Data')
plt.xlabel('Day')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Google closing prices, smooth_type = 'median', window_size = 150

smoothed_google_closing  = smooth_data(google_np_closing, smooth_type = 'median', window_size = 150)

# plot smoothed_google_closing on top of google_np_closing


plt.figure(figsize=(14, 7))

# Original Google closing prices
plt.plot(google_np_closing, label='Original Google Closing Prices', linestyle='-')

# Smoothed Google closing prices
plt.plot(smoothed_google_closing, label='Smoothed Google Closing Prices (Mean, Window=150)', linestyle=':')

plt.title('Google Closing Prices and Smoothed Data')
plt.xlabel('Day')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Dow Jones Index closing prices, smooth_type = 'mean', window_size = 200

smoothed_dji_closing  = smooth_data(DJI_np_closing, smooth_type = 'mean', window_size = 200)

# plot smoothed_dji_closing on top of dji_np_closing
# Plotting the original and smoothed data
plt.figure(figsize=(14, 7))

# Original DJI closing prices
plt.plot(DJI_np_closing, label='Original DJI Closing Prices', linestyle='-')

# Smoothed DJI closing prices
plt.plot(smoothed_dji_closing, label='Smoothed DJI Closing Prices (Mean, Window=200)', linestyle=':')

plt.title('DJI Closing Prices and Smoothed Data')
plt.xlabel('Day')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

## Extra credit: Code efficiency
### Achieve a runtime speed of < 50ms

timeit -n 1 -r 7 smoothed_google_closing  = smooth_data(google_np_closing, smooth_type = 'median', window_size = 150)

### Exercise 4: Ranking Daily Stock Surges/Crashes

ipyimage('lab4_exercise4.PNG', width = 1000)

def detect_surge_crash(filepath, detect_type, num_output_dates):
    
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Calculate the price change (Close - Open)
    df['Price Change'] = df['Close'] - df['Open']
    
    # Sort the DataFrame based on price changes
    if detect_type == 'surge':
        sorted_df = df.sort_values(by='Price Change', ascending=False)
    elif detect_type == 'crash':
        sorted_df = df.sort_values(by='Price Change')
    else:
        print("detect_type must be 'surge' or 'crash'")
        return [], []  # Return empty lists if detect_type is incorrect
    
    # Select the top events based on num_output
    top_events = sorted_df.head(num_output_dates)
    
    # Extract the dates and price changes
    date_list = top_events['Date'].tolist()
    price_change_list = top_events['Price Change'].tolist()
    
    return date_list, price_change_list

date_list_t, price_change_list_t = detect_surge_crash(filepath = "C:/Users/ankit/Downloads/Lab4_Report_Template/Lab4_Report_Template/TSLA.csv", detect_type = 'surge', num_output_dates = 5)

print(date_list_t, price_change_list_t)

date_list_g, price_change_list_g = detect_surge_crash(filepath = "C:/Users/ankit/Downloads/Lab4_Report_Template/Lab4_Report_Template/GOOGL.csv", detect_type = 'crash', num_output_dates = 7)

print(date_list_g, price_change_list_g)

## Extra credit: Code efficiency
### Achieve a runtime speed of < 10ms

timeit -n 1 -r 7 date_list_t, price_change_list_t = detect_surge_crash(filepath = "C:/Users/ankit/Downloads/Lab4_Report_Template/Lab4_Report_Template/TSLA.csv", detect_type = 'surge', num_output_dates = 5)

### Exercise 5: Human Debugger

ipyimage('lab4_exercise5.PNG', width = 1000)

### Faulty function #1 

def average_data_per_col(arr_2d):
    
    # NOTE FROM YOUR FRIEND PREPARING FOR STARBUCKS SOFTWARE ENGINEER TECH INTERVIEW 
    """  The function takes numpy 2d array as an input, computes mean for each column data, and outputs 1D array 
         with the length equal to the # of columns.
         
         For some reason I keep getting errors.... I need your help to debug the code.
         I need this position so I can get free ice lattes... :(
    
    """
    
    # placeholder for averaged values
    averaged_data = ()
    
    # Loop through each column data to compute mean and append to averaged_data 
    for k in range(len(arr_2d[:, 0])):
        
        averaged_column_data = np.mean(arr_2d[:, k])
        averaged_data.append(averaged_column_data)
        
    # Return numpy array form of the averaged_data
    return np.array(averaged_data)

# Load diabetes.csv and convert to numpy array

diabetes = pd.read_csv('diabetes.csv')
diabetes_np = diabetes.to_numpy()

# Run faulty function 1

averaged_diabetic_attributes =  average_data_per_col(diabetes_np)

def average_data_per_col_fixed(arr_2d):
    
    # Fixed placeholder for averaged values
    averaged_data = []  # Use a list instead of a tuple
    
    # Fixed loop to iterate over columns instead of rows
    for k in range(arr_2d.shape[1]):  # shape[1] gives the number of columns
        
        averaged_column_data = np.mean(arr_2d[:, k])
        averaged_data.append(averaged_column_data)
        
    # Return numpy array form of the averaged_data
    return np.array(averaged_data)

# Load diabetes.csv and convert to numpy array

diabetes = pd.read_csv('diabetes.csv')
diabetes_np = diabetes.to_numpy()

# Run the fixed function

averaged_diabetic_attributes =  average_data_per_col_fixed(diabetes_np)


# Test your fixed function

averaged_diabetic_attributes =  average_data_per_col_fixed(diabetes_np)

# Compare with correct results

correct_result_func1 = np.load('E5_correct_output_1.npy')

# Should return True if the result is correct
np.sum(np.round(correct_result_func1, 3) == np.round(averaged_diabetic_attributes, 3)) == len(correct_result_func1) 

### Faulty function #2

def daily_stock_change_2_normalized_percentage(opening_price_arr, closing_price_arr):
    
    # NOTE FROM YOUR FRIEND WHO INVESTED IN TESLA
    """  I want to write a function which takes 2 1D numpy arrays of each corresponding to opening/closing prices of stock
         and output 1D array of daily stock change in percentages. 
         
         I want the percentages scale to be in a scale such that 1 = 100%, -0.5 = -50%, 1.5 = 150%  etc.
         For example, day 1 opening: $10, day 1 closing: $15, change in scaled percecntage: 0.5.
         
         I am not really getting errors but the numbers don't look right... Can you help me?? :'( 
    
    """
    
    # placeholder for change percentage values
    change_percentages = np.zeros(len(opening_price_arr), dtype = 'int')
    
    # Loop through each opening/closing price to compute the change percentage
    for date_num in range(len(opening_price_arr)):
        
        change_percentages[date_num] = opening_price_arr[date_num] - closing_price_arr[date_num] / opening_price_arr[date_num]
    
    return change_percentages

# Load tsla.csv and convert to numpy array

tesla = pd.read_csv('TSLA.csv') 
tesla_np = tesla.to_numpy()

# Run faulty function 2

change_percentages = daily_stock_change_2_normalized_percentage(tesla_np[:, 1], tesla_np[:, 4])
print(change_percentages)

def daily_stock_change_2_normalized_percentage_fixed(opening_price_arr, closing_price_arr):
    
     # Changed dtype to 'float' to allow decimal numbers
    change_percentages = np.zeros(len(opening_price_arr), dtype='float')
    
    # Fixed the calculation of change percentage
    for date_num in range(len(opening_price_arr)):
        # Correct order of operations with parentheses
        change = (closing_price_arr[date_num] - opening_price_arr[date_num]) / opening_price_arr[date_num]
        change_percentages[date_num] = change  # No need to scale as 1 is already 100%

    
    return change_percentages

# Test your fixed function

change_percentages = daily_stock_change_2_normalized_percentage_fixed(tesla_np[:, 1], tesla_np[:, 4])
print(change_percentages)

# Compare with correct results

correct_result_func2 = np.load('E5_correct_output_2.npy')

# Should return True if the result is correct
np.sum(np.round(correct_result_func2, 3) == np.round(change_percentages, 3)) == len(correct_result_func2) 

### Faulty function #3

def subset_diabetes_by_age(diabetes_data):
    
    # NOTE FROM YOUR FRIEND WHO WORKS AT UW IHME
    """ The function takes diabetes pandas data frame as an input and outputs a subplot of 3 x 1 with three histograms.    
    
        Specifically, I want to divide the diabetes data into three age groups - 
            1. 20 to 40
            2. 40 to 60
            3. 60 to 80
            
        and plot 3 histograms of glucose distribution (50 bins per histogram) in 3 x 1 python subplots.
        
        I seem to be getting error from very beginning even before I divide the dataset....
        My coworkers at IHME prefer R rather than Python... so you are my only hope!  
    
    """
    
    # Extract the age column of the diabetes_data 
    age_column = diabetes_data[:, 8] 
    
    # Construct boolean mask for each age group 
    age_20_40_bool_mask = age_column > 20 + age_column < 40
    age_40_60_bool_mask = age_column > 40 + age_column < 60
    age_60_80_bool_mask = age_column > 60 + age_column < 80
    
    # Get glucose data for each age group via applying the boolean mask for each age group
    age_20_40_glucose = diabetes_data[age_20_40_bool_mask, 2]
    age_40_60_glucose = diabetes_data[age_40_60_bool_mask, 2]
    age_60_80_glucose = diabetes_data[age_60_80_bool_mask, 2]
    
    # Plot the histogram for each age group in 3 x 1 subplot
    fig = plt.figure(figsize=(15,7))
    
    plt.subplot(3,1,1)
    
    plt.hist(age_20_40_glucose, bins = 50)
    plt.title('Age 20 to 40', fontsize = 15)
    
    plt.subplot(3,1,2)
    
    plt.hist(age_40_60_glucose, bins = 50)
    plt.title('Age 40 to 60', fontsize = 15)
    
    plt.subplot(3,1,3)
    
    plt.hist(age_60_80_glucose, bins = 50)
    plt.title('Age 60 to 80', fontsize = 15)
    
    plt.tight_layout()
    
    plt.show()

# Load diabetes.csv as pandas dataframe

diabetes = pd.read_csv('diabetes.csv')

# Run faulty function 3

subset_diabetes_by_age(diabetes_data = diabetes)

def subset_diabetes_by_age_fixed(diabetes_data):
    
    # Ensure diabetes_data is a DataFrame
    if not isinstance(diabetes_data, pd.DataFrame):
        raise ValueError("The input data must be a pandas DataFrame.")

    # Assuming 'Age' and 'Glucose' are column names in the dataset
    age_20_40 = diabetes_data[(diabetes_data['Age'] > 20) & (diabetes_data['Age'] < 40)]
    age_40_60 = diabetes_data[(diabetes_data['Age'] >= 40) & (diabetes_data['Age'] < 60)]
    age_60_80 = diabetes_data[(diabetes_data['Age'] >= 60) & (diabetes_data['Age'] <= 80)]

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    axs[0].hist(age_20_40['Glucose'], bins=50, color='blue', alpha=0.7)
    axs[0].set_title('Age 20 to 40')
    
    axs[1].hist(age_40_60['Glucose'], bins=50, color='orange', alpha=0.7)
    axs[1].set_title('Age 40 to 60')
    
    axs[2].hist(age_60_80['Glucose'], bins=50, color='green', alpha=0.7)
    axs[2].set_title('Age 60 to 80')
    
    plt.tight_layout()
    plt.show()


# Test your fixed function
# Compare your plot with the correct plot provided in template folder

subset_diabetes_by_age_fixed(diabetes_data = diabetes)

