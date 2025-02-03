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
    gs  = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

    inc_hosps      = pd.read_csv("./data_sets/target-hospital-admissions.csv")

    #--ILI
    #--ILI data
    ili_data = pd.read_csv("./data_sets/ilidata.csv")
    ili_data["week"] = [ int(str(x)[-2:]) for x in ili_data.epiweek]

    #--lab data
    lab_data = pd.read_csv("./data_sets/clinical_and_public_lab_data__formatted.csv")

    ili_augmented = lab_data.merge(ili_data, on = ["state","epiweek","week"])

    #--CUT ALL OF THIS DATA TO BEFORE THE BEGINNING OF LAST SEASON.
    start_of_2324_season = Week(2023,40).startdate().strftime("%Y-%m-%d")
    inc_hosps            = inc_hosps.loc[ inc_hosps.date < start_of_2324_season]
    ili_augmented        = ili_augmented.loc[ ili_augmented.epiweek <202340 ]


    ili_augmented                   = ili_augmented.loc[ili_augmented.region_type=="National"] 
    ili_augmented["num_ili_scaled"] = ili_augmented.num_ili*(ili_augmented.percent_positive/100)
    
    ili_augmented["prop"] = 100*ili_augmented["num_ili_scaled"] / ili_augmented["num_patients"]

    week_2_modelweek = { y:x for x,y in zip( np.arange(32+1), list(np.arange(40,52+1)) + list(np.arange(1,20+1))  )  }
    
    ili_augmented["modelweek"] = [ week_2_modelweek[int(x)] if x in week_2_modelweek else -1 for x in ili_augmented.week.values]

    ili_augmented = ili_augmented.loc[ili_augmented.modelweek>-1] #--remove offseason

    colors = sns.color_palette("tab10",3)
    
    ax = fig.add_subplot(gs[0, 0])
    sns.lineplot(x="modelweek",y="prop",hue="season",data=ili_augmented, palette=[colors[0]],alpha=0.90,ax=ax)
    ax.legend_.remove()

    ax.set_xlabel("Epidemic week")
    ax.set_ylim(0,2.5)

    ax.set_xticks([0,10,20,32])
    ax.set_xticklabels(['40','50','08','20'])
    ax.set_ylabel("Influenza-like illness\n(ILI-NET)")
    
    ax.text(0.95,0.95,s="B.",fontweight="bold",fontsize=15,transform=ax.transAxes,ha="right",va="top",color="black")   

    #--MMWR
    mmwr_ve = pd.read_csv("./data_sets/VE_mmwr.csv")
    ax = fig.add_subplot(gs[0, 1])

    sns.barplot(x="season", y= "ve", data = mmwr_ve, ax=ax, order=["{:d}/{:d}".format(x,x+1) for x in np.arange(2015,2024+1) ])
    ax.set_yticks([0,0.25,0.50,0.75])
    ax.set_yticklabels(["0%","25%","50%","75%"])
    ax.set_ylabel("Vaccine Efficacy\n(MMWR)")

    ax.set_xlabel("")
    ax.set_xticks( [0,2,4,6,8] )
 
    ax.set_xticklabels( ["2015/16","17/18","19/20","21/22","23/24"] , rotation=45, ha="right", fontsize=10 )
    
    #ax.invert_xaxis()

    ax.text(0.95,0.95,s="C.",fontweight="bold",fontsize=15,transform=ax.transAxes,ha="right",va="top",color="black")
    
    #--NOAA
    weather_data = pd.read_csv("./data_sets/weekly_weather_data.csv")
    US_weather   = weather_data.loc[weather_data.location=="US"]

    time_data = ili_augmented[["location","season","year","week"]].drop_duplicates()
    time_data["end_date"] = [datetime.strftime( Week(row.year,row.week).enddate()  ,"%Y-%m-%d") for idx, row in time_data.iterrows()]
 
    US_weather = US_weather.merge( time_data
                                   , left_on  = ["location","enddate","year","week"]
                                   , right_on = ["location","end_date","year","week"] )

    from_week_to_model_week = pd.DataFrame({ "week": list(np.arange(40,52+1)) + list(np.arange(1,20+1)), "model_week":np.arange(33)  })

    US_weather = US_weather.merge(from_week_to_model_week, on =["week"])

    def smooth(x,y):
        from scipy.ndimage import gaussian_filter1d
        smoothed_signal = gaussian_filter1d(x[y], 2)
        x["{:s}_smooth".format(y)] = smoothed_signal
        return x
    US_weather = US_weather.groupby(["season"]).apply( lambda x: smooth(x,"tavg")  ).reset_index(drop=True)
    US_weather = US_weather.groupby(["season"]).apply( lambda x: smooth(x,"pavg")  ).reset_index(drop=True)
    
    ax = fig.add_subplot(gs[0, 2])
    sns.lineplot( x="model_week" , y = "tavg_smooth", hue="season", palette = [colors[1]], data = US_weather, errorbar="se", ax=ax)

    twin = ax.twinx()
    sns.lineplot( x="model_week" , y = "pavg_smooth", hue="season", data = US_weather, palette = [colors[2]], ax=twin)
    twin.legend_.remove()
    ax.legend_.remove()
    
    ax.text(0.95,0.95,s="D.",fontweight="bold",fontsize=15,transform=ax.transAxes,ha="right",va="top",color="black")

    ax.set_xticks([0,10,20,32])
    ax.set_xticklabels(['40','50','08','20'])

    ax.set_xlabel("Epidemic week")
    ax.set_ylabel("Temperature (C)",labelpad=0)
    
    twin.spines["left"].set_position(("axes", -0.45))
    twin.spines["left"].set_visible(True)
    
    twin.yaxis.set_label_position('left')
    twin.yaxis.set_ticks_position('left')

    twin.set_yticks([1015,1020,1023])
    twin.set_ylabel("Pressure (mmHg)",labelpad=0)
    
    ax.yaxis.label.set_color(colors[1])
    twin.yaxis.label.set_color(colors[2])

    #--forecasts

    plt.subplots_adjust(wspace=-0.5)
    fig.set_size_inches(8.5-2, (11-2)*(1/3))

    plt.savefig("./viz/data_signals.pdf")
    plt.close()
