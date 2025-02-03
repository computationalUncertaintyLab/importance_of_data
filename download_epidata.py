#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from delphi_epidata import Epidata

if __name__ == "__main__":

    # Define the range of epiweeks (one decade: 2013w40 to 2023w39)
    start_week = 201040
    end_week   = 202539

    # List of all U.S. state codes
    states = [
        'ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'ia', 
        'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo', 'ms', 
        'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm', 'nv', 'ny', 'oh', 'ok', 'or', 'pa', 
        'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy','pr','nat'
    ]

    state_to_fips = {
        'al': '01', 'ak': '02', 'az': '04', 'ar': '05', 'ca': '06', 'co': '08',
        'ct': '09', 'de': '10', 'dc': '11', 'fl': '12', 'ga': '13', 'hi': '15',
        'id': '16', 'il': '17', 'in': '18', 'ia': '19', 'ks': '20', 'ky': '21',
        'la': '22', 'me': '23', 'md': '24', 'ma': '25', 'mi': '26', 'mn': '27',
        'ms': '28', 'mo': '29', 'mt': '30', 'ne': '31', 'nv': '32', 'nh': '33',
        'nj': '34', 'nm': '35', 'ny': '36', 'nc': '37', 'nd': '38', 'oh': '39',
        'ok': '40', 'or': '41', 'pa': '42', 'ri': '44', 'sc': '45', 'sd': '46',
        'tn': '47', 'tx': '48', 'ut': '49', 'vt': '50', 'va': '51', 'wa': '53',
        'wv': '54', 'wi': '55', 'wy': '56','pr':'72','nat':'US'
    }

    def fromepiweek_to_season(x):
        from epiweeks import Week
        x = str(x)
        
        yr,wk = int(x[:4]), int(x[-2:])
        if wk>=40 and wk<=53:
            return "{:d}/{:d}".format(yr,yr+1)
        elif wk>=1 and wk<=20:
            return "{:d}/{:d}".format(yr-1,yr)
        else:
            return "offseason"
    
    # Fetch data for all states across the decade
    result = Epidata.fluview(states, Epidata.range(start_week, end_week))

    # Check and print the response
    if result['result'] == 1:
        data = result['epidata']
        print(f"Fetched {len(data)} records.")
    else:
        print(f"Failed to fetch data: {result['message']}")

    for n,record in enumerate(data):
        fip = state_to_fips.get(record['region'])
        record["fips"]   = fip
        record["season"] = fromepiweek_to_season(record.get('epiweek'))
        
        d = pd.DataFrame({ k:[v] for k,v in record.items()})
        d = d.rename(columns = {"fips":"location","region":'state'})
        
        if n==0:
            d.to_csv("./data_sets/ilidata.csv", mode='w',header=True,index=False)
        else:
            d.to_csv("./data_sets/ilidata.csv", mode='a',header=False,index=False)
