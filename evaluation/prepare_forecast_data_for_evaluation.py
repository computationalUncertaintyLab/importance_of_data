#mcandrew

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":


    forecast_with_signals    = pd.read_csv("../forecasts/with_signals/weekly_forecasts__US__2025-02-08.csv")
    forecast_without_signals = pd.read_csv("../forecasts/without_signals/weekly_forecasts__US__2025-02-08.csv")

    #--add model column
    forecast_with_signals["model"]    = "with_signals"
    forecast_without_signals["model"] = "without_signals"


    #--add truth data
    hospitalizations = pd.read_csv("../data_sets/target-hospital-admissions.csv")
    
    forecasts = pd.concat([forecast_with_signals, forecast_without_signals])
    forecasts = forecasts.merge( hospitalizations, left_on = ["location","target_end_date"], right_on = ["location","date"]  )


    #--rename columns to align with evaluation code
    forecasts = forecasts.rename(columns = {"value_x":"predicted", "value_y":"observed","output_type_id":"quantile_level"} )

    forecasts = forecasts[ ["model","location","target","target_end_date","horizon","output_type", "quantile_level","predicted","observed"] ]
    
    forecasts.to_csv("./forecasts_formatted_for_evaluation.csv",index=False)
    

    
    

    

