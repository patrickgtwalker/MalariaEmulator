# Prevalence Prediction and Simulation Analysis

This project provides tools for analyzing malaria prevalence and associated metrics through various simulations. It includes helper functions, data processing, and machine learning models to create predictive systems using simulation data frpm malsimgem.

## Repository Overview

- A primary Jupyter Notebook: `Prevalence_prediction_2-10_and_ANC.ipynb`
- Helper scripts for sequence creation and model development:
  - `model_exp.py`
  - `sequence_creator_multi_inputs_outputs.py`
- Custom functions for preprocessing simulation data.

---

## Key Features

### Main Components
1. **Annual Averages Calculation**
   - Analyzes simulation data to compute annual averages for key metrics (e.g., true prevalence, transmission intensity) for years 2, 5, and 8.

2. **Monthly Data Filtering**
   - Extracts monthly data for years 10 to 20 from simulation results.

3. **Sequence Creation**
   - Processes time-series data to generate input-output pairs suitable for model training, handling padding for initial time steps.

4. **Machine Learning Model**
   - Implements an LSTM-based architecture to predict malaria metrics (`EIR_true`, `incall`) using simulation data.

