# MMA3001 Smart Building Sensor Reconstruction

## Project Overview

This project investigates how accurately missing temperature measurements from the Monash Smart Infrastructure Lab environmental sensor dataset can be reconstructed.

Three reconstruction approaches were compared:

- Linear interpolation
- Linear regression
- Decision tree regression

The project also investigates how reconstruction performance changes with missing-data gap size and compares the computational cost of the different methods.

## Dataset

The dataset contains environmental measurements collected from the Monash Smart Infrastructure Lab, including:

- Pressure (mb)
- Temperature (°C)
- Humidity (RH %)
- CO₂ (ppm)
- LVOC (ppb)
- Formaldehyde (µg/m³)
- Timestamp

Temperature is the target variable reconstructed in this project.

For the machine-learning models, the final predictor set includes:

- Pressure
- Humidity
- CO₂
- Cyclical hour-of-day features (`hour_sin` and `hour_cos`)

The dataset used in the project is included in the `data/` directory.

## Computational Methods

### Linear Interpolation

Linear interpolation estimates missing temperatures using known temperature measurements immediately before and after a missing point.

It is used as the main numerical baseline.

### Linear Regression

Linear regression predicts temperature from the available environmental and time-related predictor variables.

### Decision Tree Regression

Decision tree regression provides a nonlinear machine-learning alternative.

The selected decision tree uses:

- `max_depth = 4`
- `min_samples_leaf = 30`
- `random_state = 42`

These parameters were selected using chronological development and validation data.

## Validation

Known temperature measurements were artificially withheld so that reconstructed values could be compared against known ground truth.

Performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The final evaluation used a chronologically later test period.

Final results were approximately:

| Method | MAE (°C) | RMSE (°C) |
|---|---:|---:|
| Linear Interpolation | 0.117 | 0.256 |
| Linear Regression | 0.643 | 0.845 |
| Decision Tree Regression | 0.811 | 1.058 |

Linear interpolation achieved the lowest reconstruction error for isolated missing temperature measurements.

## Engineering Recommendation

Linear interpolation is recommended for retrospective reconstruction of short missing temperature intervals when valid temperature measurements are available immediately before and after the missing region.

Linear regression may provide an alternative when neighbouring temperature measurements are unavailable but other environmental measurements remain available.

However, in this dataset, the tested regression models did not outperform interpolation.

## Repository Structure

```text
MMA3001-smart-building-sensor-reconstruction/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── 01_data_exploration.ipynb
├── 02_reconstruction_validation_ipynb.ipynb
│
├── data/
│   └── Env_data_MSI_Lab - 30-07-2025_19-12-2025.csv
│
├── report/
│   └── MMA3001 Project Report_submission_final.pdf
│
├── src/
│   ├── __init__.py
│   └── reconstruction.py
│
├── tests/
│   ├── test_reconstruction.py
│   └── test_report.txt
│
└── docs/
    └── HTML source-code documentation
```

## Source Code

Reusable computational functions are contained in:

```text
src/reconstruction.py
```

These include functions for:

- calculating MAE and RMSE
- linear interpolation
- training linear regression models
- training decision tree regression models
- generating cyclical time features

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Running the Tests

Automated tests are contained in:

```text
tests/test_reconstruction.py
```

Run all tests using:

```bash
python -m pytest -v
```

The current test suite checks:

- MAE and RMSE calculation
- linear interpolation using a known midpoint
- linear regression model output
- decision tree model output
- cyclical time-feature calculations

The latest test results are recorded in:

```text
tests/test_report.txt
```

## Viewing the HTML Documentation

The source-code documentation was generated from Python docstrings using `pdoc`.

To view it locally:

```bash
python -m http.server 8000 --directory docs
```

Then open port `8000` in a web browser.

## Project Report

The final project report is included in the:

```text
report/
```

directory.

The report documents the engineering problem, dataset, computational methods, validation, results, optimisation, limitations, lessons learned and AI-use reflection.

## Main Limitations

- Sensor measurements are irregularly sampled in time.
- Linear interpolation requires known temperature measurements on both sides of the missing interval.
- The interpolation approach is therefore mainly suitable for retrospective reconstruction rather than real-time prediction.
- Regression models assume that environmental predictor variables remain available when temperature is missing.
- Missing-gap size measured in number of observations does not necessarily represent a fixed elapsed time because of irregular sampling.
- Results are specific to the available dataset and should not be assumed to generalise to all buildings or sensor systems.

## Testing and Reproducibility

The repository includes:

- the original project dataset
- reproducible Jupyter notebooks
- reusable Python source code
- automated `pytest` tests
- a saved test report
- dependency information in `requirements.txt`
- generated HTML source-code documentation
- the final project report

Version control is maintained using Git and GitHub.
