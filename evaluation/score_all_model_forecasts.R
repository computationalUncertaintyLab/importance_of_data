#mcandrew;

library(scoringutils)
library(magrittr)
library(dplyr)

forecasts  = read.csv("./forecasts_formatted_for_evaluation.csv")
forecasts$predicted = as.numeric(forecasts$predicted)

forecasts$quantile_level = as.numeric(forecasts$quantile_level)
forecasts$quantile_level = round(forecasts$quantile_level,3)

model_forecast  = as_forecast_quantile(data = forecasts
                             ,forecast_unit = c("model", "target_end_date", "horizon" ,"location")
                             ,forecast_type = "quantile")
all_scores     = model_forecast |> score()
write.csv(all_scores, "./evaluation_metrics.csv")
