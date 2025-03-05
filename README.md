# Prevalence Prediction and Simulation Analysis

This project aims to provide tool for estimating malaria incidence and transmission intensity through emulation of timeseries prevalence data. It includes helper functions, data processing, and machine learning models to create predictive systems using simulation data frpm malsimgem.

## Repository Overview

- A primary Jupyter Notebook: `ANC_Emulator_PyTorch.ipynb`
- Helper scripts for sequence creation and model development:
  - `model_exp.py`
  - `sequence_creator.py`
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


## Folder Structure

```
project_root/
│── src/                  # Source code/functions directory
│   ├── preprocessing.py  # Handles data loading and preprocessing
│   ├── sequence_creator.py  # Creates birectional sequences of tensors
│   ├── model_exp.py      # Defines PyTorch model and training
│   ├── test.py           # Model testing
│   ├── utils.py          # Utility functions (metrics, visualization, etc.)
|   ├── inference.py           
│── test/                # Unit tests
│   ├── test_data         # Aim to test model under various conditions - seasonality, randomwalk, routine data...
│── notebooks/            # Jupyter notebooks 
│── requirements.txt      # Dependencies
│── README.md             # Project overview and instructions
│── .gitignore            # Ignored files (e.g., __pycache__, .venv)
│── setup.py              # Setup script (to be updated)
│── pyproject.toml        # Modern package management (to be updated)
├── ANC_Emulator_PyTorch.ipynb  # Main analysis notebook
├── data/                  # simulated data from mamasante/malsimgem
├── sequence_creator_multi_inputs_outputs.py  # Sequence generation for ML training
├── plots/                 # Saved visuals

