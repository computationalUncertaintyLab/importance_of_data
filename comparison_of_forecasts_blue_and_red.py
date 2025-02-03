#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns

from datetime import datetime, timedelta

from epiweeks import Week

import scienceplots

if __name__ == "__main__":

    plt.style.use("science")
    
    fig = plt.figure(constrained_layout=True)
    gs  = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    forecasts_with_signals    = pd.read_csv("./forecasts/with_signals/all_forecasts__US__2025-02-08.csv")
    forecasts_without_signals = pd.read_csv("./forecasts/without_signals/all_forecasts__US__2025-02-08.csv") #time stamp refers to when this was generated

    forecasts = forecasts_with_signals
    forecasts = pd.pivot_table(index="target_end_date", columns = ["output_type_id"], values = ["value"], data = forecasts_with_signals)
    forecasts.columns= [ y for x,y in forecasts.columns]
    
    #forecasts.index = [ datetime.strptime(x,"%Y-%m-%d") for x in forecasts.index.values]
    forecasts = forecasts.reset_index()

    def from_date_to_epiweek(row):
        from epiweeks import Week
        ew = Week.fromdate( datetime.strptime(row.target_end_date,"%Y-%m-%d") )

        row["year"] = ew.year
        row["week"] = ew.week

        return row
    from_week_to_model_week = pd.DataFrame({ "week": list(np.arange(40,52+1)) + list(np.arange(1,20+1)), "model_week":np.arange(33)  })


    forecasts = forecasts.apply(from_date_to_epiweek,1 )
    forecasts = forecasts.merge(from_week_to_model_week, on = ["week"])

    ax = fig.add_subplot(gs[0,0])

    ax.fill_between( forecasts.model_week.values, forecasts['0.025'] , forecasts['0.975'], color="blue", alpha = 0.125  )
    ax.fill_between( forecasts.model_week.values, forecasts['0.250'] , forecasts['0.750'], color="blue", alpha = 0.125  )
    ax.plot(         forecasts.model_week.values, forecasts['0.500'] , color="blue", alpha=0.95, label="Model with all data sources" )
    ax.scatter(      forecasts.model_week.values, forecasts['0.500'] , color="blue" ,s=5, alpha=0.99)


    forecasts = forecasts_without_signals
    forecasts = pd.pivot_table(index="target_end_date", columns = ["output_type_id"], values = ["value"], data = forecasts_without_signals)
    forecasts.columns= [ y for x,y in forecasts.columns]
    forecasts = forecasts.reset_index()

    forecasts = forecasts.apply(from_date_to_epiweek,1 )
    forecasts = forecasts.merge(from_week_to_model_week, on = ["week"])
    
    #forecasts.index = [ datetime.strptime(x,"%Y-%m-%d") for x in forecasts.index.values]
    
    ax.fill_between( forecasts.model_week.values, forecasts['0.025'] , forecasts['0.975'], color="red", alpha = 0.125  )
    ax.fill_between( forecasts.model_week.values, forecasts['0.250'] , forecasts['0.750'], color="red", alpha = 0.125  )
    ax.plot(         forecasts.model_week.values, forecasts['0.500'] , color="red", alpha=0.95, label="Model with only NHSN data" )
    ax.scatter(      forecasts.model_week.values, forecasts['0.500'] , color="red" ,s=5, alpha=0.99)

    ax.legend(loc="right", frameon=False,fontsize=11,markerscale=2, labelspacing=0.1)
    
    location       = "US"
    inc_hosps      = pd.read_csv("./data_sets/target-hospital-admissions.csv")
    location_hosps = inc_hosps.loc[inc_hosps.location==location]

    def datetime_to_season(x):
        from epiweeks import Week
        from datetime import datetime
        
        x = str(Week.fromdate( datetime.strptime(x.date,"%Y-%m-%d") ).cdcformat())

        yr,wk = int(x[:4]), int(x[-2:])
        if wk>=40 and wk<=53:
            return "{:d}/{:d}".format(yr,yr+1)
        elif wk>=1 and wk<=20:
            return "{:d}/{:d}".format(yr-1,yr)
        else:
            return "offseason"

    def add_epiweek_data(x):
        from epiweeks import Week
        from datetime import datetime
        
        ew           = Week.fromdate( datetime.strptime(x.date,"%Y-%m-%d") )
        x["week"]    = ew.week
        x["year"]    = ew.year
        x["epiweek"] = ew.cdcformat()

        return x
        
    location_hosps["season"] = location_hosps.apply(datetime_to_season,1)
    location_hosps           = location_hosps.apply(add_epiweek_data  ,1) 
    
    location_hosps           = location_hosps.loc[location_hosps.season!="offseason"]

    #--reorder weeks
    weeks = pd.DataFrame({"week": list(np.arange(40,52+1)) + list(np.arange(1,20+1)) })
    
    location_hosps__2023     = location_hosps.loc[location_hosps.season=="2023/2024"]
    location_hosps__2023     = weeks.merge(location_hosps__2023, on = "week") 
    
    location_hosps__2022     = location_hosps.loc[location_hosps.season=="2022/2023"]
    location_hosps__2022     = weeks.merge(location_hosps__2022, on = "week") 

    location_hosps__2021     = location_hosps.loc[location_hosps.season=="2021/2022"]
    location_hosps__2021     = weeks.merge(location_hosps__2021, on = "week") 
    
    ax.plot(forecasts.index.values, location_hosps__2023.value.values, color="black",lw=2)
    ax.plot(forecasts.index.values, location_hosps__2022.value.values, color="0.50")
    ax.plot(forecasts.index.values, location_hosps__2021.value.values, color="0.50")

    ax.set_ylim(0,30000)

    ax.set_yticks([5000,10000,20000,30000])
    ax.set_yticklabels(["5k","10k","20k","30k"])
    
    ax.set_xticks([0,10,20,32])
    ax.set_xticklabels(['40','50','08','20'])

    ax.set_xlabel("Epidemic week")
    
    ax.set_ylabel("US Influenza Hospitalizations")

    ax.set_xlim(0,32)
    
    fig.set_size_inches(8.5-2, (11-2)*(1/3))
    
    plt.savefig("./viz/comparison_forecasts.pdf")
