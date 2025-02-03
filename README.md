# When Data Disappear: The Cost of Policy on Public Health

All code to download the data, run the forecasts, and produce the figure in this manuscript are contained in a Makefile. 
The entire pipeline can be run by typing `make` in the terminal. 

## Data Sets 

**Clinical Lab data from ILI-NET** 
1. Code to download this data is `./download_lab_percentage_data.R`.
2. Data is downloaded using the cdcfluview package
The weekly percent positive influenza cases at state and US national level are collected from 2015 to present.
The dataset that is produced is called `./data_sets/clinical_and_public_lab_data.csv`. 

**Clinical Lab data from ILI-NET** 
1. Code to download this data is `./download_epidata.py`.
2. Data is downloaded using the delphi-epidata package

