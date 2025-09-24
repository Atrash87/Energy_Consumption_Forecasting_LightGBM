# Electricity Consumption Dataset

This dataset contains electricity consumption data for 370 clients, recorded every 15 minutes.  

## Dataset Characteristics
- **Type:** Time-Series  
- **Subject Area:** Computer Science  
- **Associated Tasks:** Regression, Clustering  
- **Feature Type:** Real  
- **Instances:** 370  
- **Features:** 140,256  

## Dataset Information
- No missing values.  
- Consumption values are in **kW**. To convert to **kWh**, divide by 4.  
- Each column represents one client. Some clients were created after 2011; in these cases, consumption is 0.  
- All timestamps are in Portuguese time.  
  - **March** (23-hour day): values between 1:00–2:00 AM are 0.  
  - **October** (25-hour day): values between 1:00–2:00 AM aggregate two hours.  

## Variable Information
- Data is saved as a CSV with **semicolon (;)** separators.  
- The first column is `DateTime` in `'yyyy-mm-dd hh:mm:ss'` format.  
- Remaining columns are float values representing consumption in kW.  

## The row Data can be found in the following link:
https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
