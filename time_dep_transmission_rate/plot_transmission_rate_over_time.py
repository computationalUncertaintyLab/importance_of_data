#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

import matplotlib.lines as mlines
import scienceplots

if __name__ == "__main__":

    with_signals    = pickle.load(open("./time_dep_transmission_rate/with_signals_posterior_samples.pkl","rb"))
    without_signals = pickle.load(open("./time_dep_transmission_rate/without_signals_posterior_samples.pkl","rb"))

    t_rate_last_season_with_signals    = with_signals["transmission_rate"][:,0,-1,1:,0]
    time_dep_trate={"value":[],"week":[]}
    for sample in t_rate_last_season_with_signals:
        repo_number = np.array(sample)*(1 + 0.40)*(3./7) #-- 0.40 is the mean vaccine effectivness and the infecitous period is assumed 3/7 of a week
        time_dep_trate["value"].extend( np.array(repo_number))
        time_dep_trate["week"].extend( np.arange(33))
    time_dep_trate_with = pd.DataFrame(time_dep_trate)
    time_dep_trate_with["model"] = "with"

    t_rate_last_season_without_signals    = without_signals["transmission_rate"][:,0,-1,1:,0]
    time_dep_trate={"value":[],"week":[]}
    for sample in t_rate_last_season_without_signals:

        repo_number = np.array(sample)*(1 + 0.40)*(3./7) #-- 0.40 is the mean vaccine effectivness and the infecitous period is assumed 3/7 of a week
        time_dep_trate["value"].extend( np.array(repo_number))
        time_dep_trate["week"].extend( np.arange(33))
    time_dep_trate_without = pd.DataFrame(time_dep_trate)
    time_dep_trate_without["model"] = "without"

    time_dep_transmission = pd.concat([time_dep_trate_with, time_dep_trate_without])

    plt.style.use("science")
    
    fig,ax = plt.subplots()
    sns.lineplot(x = "week", y="value", hue="model", data = time_dep_transmission,ax=ax, palette = ["blue","red"])

    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("Time dependent\neffective reproduction\nnumber")

    blue_line = mlines.Line2D([], [], color='blue', label='US model with all data sources')
    red_line = mlines.Line2D([], [], color='red'  , label='US model without data sources')

    ax.legend(handles=[blue_line, red_line] ,fontsize=9, frameon=False)

    fig.set_tight_layout(True)
    fig.set_size_inches(8.5-2, (11-2)/4)
    
    plt.savefig("./time_dep_transmission_rate/time_depedent_transmission_rate.pdf")
    plt.close()
