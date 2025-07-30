# Prevalence Prediction and Simulation Analysis

This project aims to provide tool for estimating malaria incidence and transmission intensity from timeseries prevalence data through emulation of mechanistic Malaria transmission model. It includes helper functions, data processing, and machine learning models pipeline to create predictive systems using simulation data from malsimgem.

## Repository Overview

## Installation
To install the project dependencies, run:
```
pip install -r requirements.txt
```
You can also install the package in editable mode:
```
pip install -e .
```


- A primary Jupyter Notebook: `ANC_Emulator_PyTorch.ipynb`
- Helper scripts for sequence creation and model development:
  - `model_exp.py`
  - `sequence_creator.py`
- Custom functions for preprocessing simulation data.
- Dashboard for estimating incidence and transmission intensities/EIR 'emulator.py'

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


## Folder Structure

```
project_root/
│── src/                  # Source code/functions directory
│   ├── preprocessing.py  # Handles data loading and preprocessing
│   ├── sequence_creator.py  # Creates birectional sequences of tensors
│   ├── model_exp.py      # Defines PyTorch model and training
│   ├── test.py           # Model testing
│   ├── utils.py          # Utility functions (metrics, visualization, etc.)
|   ├── inference.py      # Specifically used for inferencing within training notebook
|   ├── inference_util.py # Specifically used for dashboard inferencing

│── test/                # Unit tests
│   ├── test_data         # Contains a thousand test runs across different transmission intensities
│── notebooks/            # Jupyter notebooks (to be updated with recent experiments)
│── requirements.txt      # Dependencies
│── README.md             # Project overview and instructions
│── emulator.py           # Python script containing deployed streamlit dashboard
│── emulator_one_model.py # Emulator variant predicting with one model   
│── emulator_two_model.py # Emulator variant predicting with two models
│── .gitignore            # Ignored files (e.g., __pycache__, .venv)
│── setup.py              # Setup script
│── pyproject.toml        # Modern package management
├── ANC_Emulator_PyTorch.ipynb  # Main analysis notebook
├── data/                  # simulated data from mamasante/malsimgem
├── plots/                 # Saved visuals


https://estimatemalariaparameters.streamlit.app/

![alt text](<Screenshot (1)-2.png>)