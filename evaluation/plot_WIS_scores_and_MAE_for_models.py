#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.lines as mlines


if __name__ == "__main__":


    evals = pd.read_csv("./evaluation_metrics.csv")


    fig, axs = plt.subplots(1,2)

    # Create custom legend handles
    blue_line = mlines.Line2D([], [], color='blue', label='US model with all data sources')
    red_line = mlines.Line2D([], [], color='red'  , label='US model without data sources')
    
    ax = axs[0]
    sns.lineplot(x = "horizon"   , y= "wis", hue="model", data = evals, ax=ax, palette = ["blue","red"])
    sns.scatterplot(x = "horizon", y= "wis", hue="model", data = evals, ax=ax, palette = ["blue","red"])

    ax.set_ylabel("Weighted interval score")
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.legend_.remove()

    ax.legend(handles=[blue_line, red_line] , ncol=2, fontsize=9, bbox_to_anchor=(2.25,1.175) , frameon=False)
    

    ax = axs[1]
    sns.lineplot(x = "horizon"   , y= "ae_median", hue="model", data = evals, ax=ax, palette = ["blue","red"])
    sns.scatterplot(x = "horizon", y= "ae_median", hue="model", data = evals, ax=ax, palette = ["blue","red"])

    ax.set_ylabel("Absolute error")
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.legend_.remove()

    plt.subplots_adjust(wspace=0.4,bottom=0.2)
    fig.set_size_inches(8.5-2, (11-2)/4)
    
    plt.savefig("./forecast_evaluations.pdf")
    plt.close()
