import streamlit as st 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.sequence_creator import create_sequences
from src.model_exp import LSTMModel

# Set page configuration
st.set_page_config(
    page_title="Malaria Estimator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply a modern Seaborn theme
sns.set_style("darkgrid")  
plt.style.use("ggplot")

# Custom CSS for better UI
st.markdown("""
    <style>
        /* Customize headers */
        h1 {
            color: #FF4B4B;
            text-align: center;
        }
        /* Improve widgets */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        /* Adjust sidebar */
        [data-testid="stSidebar"] {
            background-color: #2E3B4E;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Function to load model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=1, architecture=[256, 128, 64, 32])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Function to preprocess data
def preprocess_data(df):
    log_transform = lambda x: np.log(x + 1e-8)

    # Sanity Check for selected prevalence column
    if not pd.api.types.is_numeric_dtype(df['prev_true']):
        st.error("🚨 The selected prevalence column is invalid. It contains non-numeric values.")
        return None, False  # Return None to indicate failure
    
    has_true_values = {'EIR_true', 'incall'}.issubset(df.columns)

    if has_true_values:
        df_scaled = df[['prev_true', 'EIR_true', 'incall']].apply(log_transform)
    else:
        df_scaled = df[['prev_true']].apply(log_transform)
    
    return df_scaled, has_true_values

# Function to convert time column
def convert_time_column(df, time_column):
    try:
        if pd.api.types.is_numeric_dtype(df[time_column]):
            return df[time_column].astype(float) / 365.25  # Convert days to years
        else:
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce', format='%b-%y')  # Example: "Jan-16"
            
            if df[time_column].isna().all():
                st.error("Could not parse the time column. Ensure it's a proper date format (e.g., Jan-16).")
                return None
        
            start_year = df[time_column].dt.year.min()
            df['time_in_years'] = df[time_column].dt.year + (df[time_column].dt.month - 1) / 12 - start_year

        if len(df['time_in_years']) != len(df):
            st.error(f"🚨 Mismatch error, possibly due to invalid time column selection'{time_column}' – expected {len(df)} values but found {len(df['time_in_years'])}. "
                     "Please verify your dataset.")
            return None
        return df['time_in_years']
    
    except Exception as e:
        st.error(f"Error in converting time column: {e}")
        return None



def compute_global_yaxis_limits(data, selected_runs, run_column, window_size, model, device, has_true_values):
    log_transform = lambda x: np.log(x + 1e-8)
    inverse_log_transform = lambda x: np.exp(x) - 1e-8

    all_prev = []
    all_eir = []

    for run in selected_runs:
        run_data = data[data[run_column] == run]
        if run_data.empty:
            continue

        all_prev.extend(run_data['prev_true'].values)

        scaled_run_data, _ = preprocess_data(run_data)
        X_test_scaled, y_test_scaled = create_sequences(scaled_run_data, window_size)

        if len(X_test_scaled) == 0:
            continue

        X_test_scaled = X_test_scaled.to(device)

        with torch.no_grad():
            test_predictions_scaled = model(X_test_scaled.unsqueeze(-1)).cpu().numpy()

        test_predictions_unscaled = inverse_log_transform(test_predictions_scaled)
        all_eir.extend(test_predictions_unscaled[:, 0])

    prev_min, prev_max = 0, max(all_prev) * 1.1 if all_prev else (0, 1)
    eir_min, eir_max = 0, max(all_eir) * 1.1 if all_eir else (0, 1)

    return (prev_min, prev_max), (eir_min, eir_max)


def plot_predictions(test_data, run_column, time_column, selected_runs, model, device, window_size,
                     log_eir, log_inc, log_all, has_true_values, prev_limits, eir_limits):
    log_transform = lambda x: np.log(x + 1e-8)
    inverse_log_transform = lambda x: np.exp(x) - 1e-8

    is_string_time = not pd.api.types.is_numeric_dtype(test_data[time_column])

    if is_string_time:
        time_labels = test_data[time_column].unique()
        time_values = np.arange(len(time_labels))
    else:
        time_values = test_data[time_column].astype(float) / 365.25
        time_labels = None

    num_plots = len(selected_runs)
    fig, axes = plt.subplots(num_plots, 2, figsize=(12, 5 * num_plots), sharex=True)
    if num_plots == 1:
        axes = np.expand_dims(axes, axis=0)

    colors = sns.color_palette("muted", 3)
    data_to_download = []

    # Use precomputed global y-axis limits
    prev_min, prev_max = prev_limits
    eir_min, eir_max = eir_limits

    for i, run in enumerate(selected_runs):
        run_data = test_data[test_data[run_column] == run]

        if run_data.empty:
            st.warning(f"Invalid run column selected: {run}")
            continue

        scaled_run_data, _ = preprocess_data(run_data)
        X_test_scaled, y_test_scaled = create_sequences(scaled_run_data, window_size)

        if len(X_test_scaled) == 0:
            st.warning("Select valid run and/or time column where applicable.")
            return

        X_test_scaled = X_test_scaled.to(device)

        with torch.no_grad():
            test_predictions_scaled = model(X_test_scaled.unsqueeze(-1)).cpu().numpy()

        test_predictions_unscaled = inverse_log_transform(test_predictions_scaled)
        X_test_unscaled = inverse_log_transform(X_test_scaled.cpu().numpy())
        time_values_plot = time_values[:len(test_predictions_scaled)]

        if has_true_values:
            y_test_unscaled = inverse_log_transform(y_test_scaled.numpy())

        titles = ["Prevalence", "EIR"]
        predictions = [X_test_unscaled[:, -1], test_predictions_unscaled[:, 0]]
        true_values = [None, y_test_unscaled[:, 0] if has_true_values else None]
        log_scales = [log_all, log_eir or log_all]

        for ax, title, color, pred, true_val, log_scale in zip(axes[i], titles, colors, predictions, true_values, log_scales):
            if title == "Prevalence":
                prev_true_adjusted = run_data['prev_true'].values[:len(pred)]
                ax.plot(time_values[:len(prev_true_adjusted)], prev_true_adjusted, linestyle="--", color=color, label="Prevalence", linewidth=2.5)
            else:
                ax.plot(time_values_plot, pred, linestyle="--", color=color, label=f"Estimated {title}", linewidth=2.5)

            if true_val is not None:
                ax.plot(time_values_plot, true_val, color="black", linestyle="-", label=f"True {title}", linewidth=2)

            if log_scale:
                ax.set_yscale('log')

            # ⬇️ Apply consistent y-axis limits
            if title == "Prevalence":
                ax.set_ylim(prev_min, prev_max)
            elif title == "EIR":
                ax.set_ylim(eir_min, eir_max)

            ax.set_title(f"{run} - {title}", fontsize=14, color="#FF4B4B")
            ax.set_ylabel(title, fontsize=12)
            ax.legend()

        data_to_download.append(pd.DataFrame({
            "Prevalence": predictions[0],
            "Estimated EIR": predictions[1]
        }))

    # ⬇️ Final axis formatting
    for ax in axes[-1]:
        if is_string_time:
            tick_indices = np.arange(0, len(time_values_plot), step=6, dtype=int)
            ax.set_xticks(time_values_plot[tick_indices])
            ax.set_xticklabels(np.array(time_labels)[tick_indices], rotation=45, fontsize=10)
        else:
            ax.set_xlabel("Years", fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)

    if data_to_download:
        combined_data = pd.concat(data_to_download, keys=selected_runs, names=[run_column, "Index"])
        csv_data = combined_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Estimates as CSV", data=csv_data, file_name="predictions.csv", mime="text/csv")

def adjust_trailing_zero_prevalence(df, prevalence_column='prev_true', min_val=0.0001, max_val=0.0009, seed=None):
    df = df.copy()
    zeros_mask = df[prevalence_column] == 0
    num_zeros = zeros_mask.sum()

    if num_zeros > 0:
        #st.warning(f"⚠️ Found {num_zeros} zero prevalence value(s); replacing with random values between {min_val} and {max_val}.")
        rng = np.random.default_rng(seed)  # Create random generator with optional seed
        random_values = rng.uniform(min_val, max_val, size=num_zeros)
        df.loc[zeros_mask, prevalence_column] = random_values
    return df

# Streamlit UI
st.title("🔬 Malaria Incidence and EIR Estimator with AI")



# Choose data source
data_source = st.radio("📊 Select data source", ("Upload my own data", "Use preloaded test data"))

# Load the data accordingly
if data_source == "Use preloaded test data":
    remote_url = "https://raw.githubusercontent.com/Olatundemi/MalariaEmulator/main/test/ANC_Simulation_1000_test_runs.csv"

    try:
        test_data = pd.read_csv(remote_url)
        st.success("✅ Preloaded test data loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load preloaded data: {e}")
        st.stop()
else:
    uploaded_file = st.file_uploader("📂 Upload prevalence data to estimate (CSV)", type=["csv"])
    if uploaded_file:
        test_data = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

columns = test_data.columns.tolist()

run_column = st.selectbox("🔄 Select geographical unit(s) e.g. Region, District, Province...", columns) if 'run' not in columns else 'run'

time_column = st.selectbox("🕒 Select time column", columns) if 't' not in columns else 't'

unique_runs = test_data[run_column].unique()
selected_runs = st.multiselect(f"📊 Select {run_column}(s) to estimate", unique_runs, default=unique_runs[:0])

# Filter the data based on selected runs
filtered_data = test_data[test_data[run_column].isin(selected_runs)]

if 'prev_true' not in columns:
    
    prevalence_column = st.selectbox("🩸 Select the column corresponding to prevalence", columns)#, key=f"prevalence_select_{key_suffix}")
    test_data = test_data.rename(columns={prevalence_column: 'prev_true'})
test_data = adjust_trailing_zero_prevalence(test_data, prevalence_column='prev_true', seed=42)


model_path = "src/trained_model/4_layers_model.pth"
window_size = 10
model, device = load_model(model_path)

df_scaled, has_true_values = preprocess_data(test_data)

if df_scaled is None:
    st.stop()  # Stop further execution if preprocessing fails
    
log_eir = st.checkbox("📈 View EIR on Log Scale", value=False)
log_inc = st.checkbox("📉 View Incidence on Log Scale", value=False)
log_all = st.checkbox("🔍 View All Plots on Log Scale", value=False)


if selected_runs:
    prev_limits, eir_limits = compute_global_yaxis_limits(
        test_data, selected_runs, run_column, window_size, model, device, has_true_values
    )
    plot_predictions(
        test_data, run_column, time_column, selected_runs, model, device, window_size,
        log_eir, log_inc, log_all, has_true_values, prev_limits, eir_limits
    )