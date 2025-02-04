#mcandrew

PYTHON ?= python3 -W ignore

VENV_DIR := .forecast
VENV_PYTHON := $(VENV_DIR)/bin/python -W ignore

R ?= Rscript

forecast: run_forecasts

download_all_data: build_env download_clinical_data download_ili download_weather_data


build_env:
	@echo "build forecast environment"
	@$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install -r requirements.txt

download_clinical_data:
	@echo "Downliading Public lab data"
	@$(R) download_lab_percentage_data.R
	@$(VENV_PYTHON) format_lab_data.py

download_ili:
	@echo "Downliading recent ili"
	@$(VENV_PYTHON) download_epidata.py

download_hosp_pct_data:
	@echo "Downloading NHSNpct hosp data"
	@$(VENV_PYTHON) download_percent_reported_hosps.py

download_weather_data:
	@echo "Download weather data"
	@$(VENV_PYTHON) download_weather_data.py

download_NSHN_data:
	@echo "Download hospitalization data"
	@$(R) get_target_data.R

run_forecasts_past_signals:
	@echo "Forecasting 2023/24 Season with Signals"
	@$(VENV_PYTHON) ./model_code/forecast_with_signals__pastseason.py

run_forecasts_past_no_signals:
	@echo "Forecasting 2023/24 Season withOUT Signals"
	@$(VENV_PYTHON) ./model_code/forecast_without_signals__pastseason.py

evaluation_metrics:
	@echo "Run evaluation metrics"
	@$(VENV_PYTHON) ./evaluation/prepare_forecast_data_for_evaluation.py
	@$(R) ./evaluation/score_all_model_forecasts.R
	@$(VENV_PYTHON) ./evaluation/plot_WIS_scores_and_MAE_for_models.py

produce_time_dep_repo_num:
	@echo "Run Time dependent repo number"
	@$(VENV_PYTHON) ./time_dep_transmission_rate/plot_transmission_rate_over_time.py

produce_figure_one_forecasts:
	@echo "Produce Fig 1 visual"
	@$(VENV_PYTHON) comparison_of_forecasts_blue_and_red.py

output_map_of_cities:
	@echo "Produce map of cities"
	@$(VENV_PYTHON) plot_of_us_cities.py

