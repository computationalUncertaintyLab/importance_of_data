# When Data Disappear: The Cost of Policy on Public Health

All code to download the data, run the forecasts, and produce the figure in this manuscript are contained in a Makefile. 
The entire pipeline can be run by typing `make` in the terminal. 

## Data Sets 

In the Makefile, users can run the command `make download_all_data` to run the three scripts that download all the data. 
If the user does not with to actively collect new data then the datasets used for analysis have already been commited and are located in the folder called `./data_sets/`

**Clinical Lab data from ILI-NET**    
1. Code to download this data is `./download_lab_percentage_data.R`.
2. Data is downloaded using the cdcfluview package
The weekly percent positive influenza cases at state and US national level are collected from 2015 to present.
The dataset that is produced is called `./data_sets/clinical_and_public_lab_data.csv`. 
This dataset is then foremmated with the code `./formatlab_data.py`. That code produces a formatted lab dataset that is used for analysis called `./data_sets/clinical_and_public_lab_data__formatted.csv`

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

**Flu Hospitalization data from NHSN**    
1. Code to download this data is `./get_target_data.R`.
2. Data is downloaded using modified code from the CDC FluSight GitHub Repository
The weekly number of incident hospitalizations at state level and a US national estimate is produced.   
The dataset that is produced is called `./data_sets/target-hospital-admissions.csv`.

**Flu Percent reported data from NHSN**
1. Code to download this data is `./download_percent_reported_hosps.py`.
2. Data is downloaded from the CDC HHS datset.
The weekly percent of all providers reporting at state level and a US national estimate is produced.   
The dataset that is produced is called `./data_sets/pct_hospital_reporting.csv`.

**Population data**   
Population data for each state was taken from the CDC hosted FluSIght GitHub repository. 
This data is a single csv, was downloaded, and storeed in `./data_sets/locations.csv`.

**Vaccine Efficacy data from MMWR reports**   
The estimated seasonal vaccine efficacy was taken from MMWR reports. 
The dataset is at `./data_sets/VE_mmwr.csv` and includes links to each MMWR report from which data was extracted. 

## Forecast (transmission model) code
The code to produce the two (red and blue) forecasts presented in the manuscript is located at `./model_code`.
These codes produce forecasts for all states and a US national forecast for the previous 2023/24 influenza season in the northern hemisphere.
The following code `./model_code/forecast_with_signals__pastseason.py` uses the above datasets to produce the blue forecast.
All forecasts for this model are stored in `./forecasts_with_signals/`.
The red forecast uses only the NHSN data and is located at `./model_code/forecast_without_signals__pastseason.py`.
All forecasts for this model are stored in `./forecasts_without_signals/`.








