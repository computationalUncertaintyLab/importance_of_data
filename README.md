# When Data Disappear: The Cost of Policy on Public Health

All code to download the data, run the forecasts, and produce the figure in this manuscript are contained in a Makefile. 
The entire pipeline can be run by typing `make` in the terminal. 

## Data Sets 

**Clinical Lab data from ILI-NET** 
1. Code to download this data is `./download_lab_percentage_data.R`.
2. Data is downloaded using the cdcfluview package
The weekly percent positive influenza cases at state and US national level are collected from 2015 to present.
The dataset that is produced is called `./data_sets/clinical_and_public_lab_data.csv`. 

**Public ILI data from ILI-NET** 
1. Code to download this data is `./download_epidata.py`.
2. Data is downloaded using the delphi-epidata package
The weekly number of patients with reported ILI and number of all patients who are reported at the hospital for all states and a US national estimate.
The dataset that is produced is called `./data_sets/clinical_and_public_lab_data.csv`. 

**GHCd data from NOAA via Meteostat** 
1. Code to download this data is `./download_weather_data.py`.
2. Data is downloaded using the meteostat package
The weekly temperature and pressure measuremnts for the three largest cities in each state in the US from 2015 to present.
In the code is a list of all the major cities and how the US average is computed.  
The dataset that is produced is called `./data_sets/weekly_weather_data.csv`. 



