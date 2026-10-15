"""
Direct Theis Parameter Recovery from MODFLOW 6 Output
=====================================================

This script estimates transmissivity (T) and storativity (S) from
MODFLOW-generated hydraulic heads using nonlinear least-squares
fitting of the full Theis solution.

Workflow
--------
1. Read MODFLOW observation heads from the H0 homogeneous model.
2. Convert hydraulic head to drawdown.
3. Restrict the fitting window to early-time data up to 1 day.
4. Fit transmissivity and storativity using the full Theis solution.
5. Calculate fitting RMSE.
6. Repeat the fitting for five observation distances and three layers.
7. Export the recovered parameters to CSV.

This forms part of the MSc project:
"Pumping-Test Simulation in MODFLOW"

Author: Rajalekshmi K Raju
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import exp1


# ---------------------------------------------------------------------
# MODEL PARAMETERS
# ---------------------------------------------------------------------

Q = 50.0               # Pumping-rate magnitude [m3/day]
INITIAL_HEAD = 10.0     # Initial hydraulic head [m]

T_TRUE = 172.8          # Reference transmissivity [m2/day]
S_TRUE = 2.0e-4         # Reference storativity [-]

DISTANCES = [50, 100, 200, 500, 1000]
NUMBER_OF_LAYERS = 3

FIT_END_TIME = 1.0      # Direct Theis fitting window [day]


# ---------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------

INPUT_FILE = Path("3L_H0_Control_Validation.xlsx")
SHEET_NAME = "H0_3L_Homo_Control.ob_gw"

OUTPUT_FILE = Path("H0_Direct_Theis_Fit.csv")


# ---------------------------------------------------------------------
# THEIS SOLUTION
# ---------------------------------------------------------------------

def theis_drawdown(time, distance, transmissivity, storativity):
    """
    Calculate analytical Theis drawdown.

    Parameters
    ----------
    time : numpy.ndarray
        Time since pumping started [day].
    distance : float
        Radial distance from the pumping well [m].
    transmissivity : float
        Aquifer transmissivity [m2/day].
    storativity : float
        Aquifer storativity [-].

    Returns
    -------
    numpy.ndarray
        Theis drawdown [m].
    """

    u = (
        distance**2 * storativity
        / (4.0 * transmissivity * time)
    )

    well_function = exp1(u)

    drawdown = (
        Q
        / (4.0 * np.pi * transmissivity)
        * well_function
    )

    return drawdown


# ---------------------------------------------------------------------
# PARAMETER FITTING
# ---------------------------------------------------------------------

def fit_theis_parameters(
    time,
    observed_drawdown,
    distance,
):
    """
    Estimate transmissivity and storativity using nonlinear
    least-squares fitting of the full Theis solution.

    The optimisation is performed in log10 parameter space so that
    transmissivity and storativity remain positive.

    Parameters
    ----------
    time : numpy.ndarray
        Pumping times [day].
    observed_drawdown : numpy.ndarray
        MODFLOW-derived drawdown [m].
    distance : float
        Observation distance from the pumping well [m].

    Returns
    -------
    fitted_T : float
        Estimated transmissivity [m2/day].
    fitted_S : float
        Estimated storativity [-].
    rmse : float
        Root Mean Square Error between fitted Theis drawdown
        and MODFLOW drawdown [m].
    """

    def residuals(log_parameters):

        transmissivity = 10 ** log_parameters[0]
        storativity = 10 ** log_parameters[1]

        predicted_drawdown = theis_drawdown(
            time=time,
            distance=distance,
            transmissivity=transmissivity,
            storativity=storativity,
        )

        return predicted_drawdown - observed_drawdown

    initial_guess = [
        np.log10(T_TRUE),
        np.log10(S_TRUE),
    ]

    result = least_squares(
        residuals,
        initial_guess,
    )

    fitted_T = 10 ** result.x[0]
    fitted_S = 10 ** result.x[1]

    rmse = np.sqrt(
        np.mean(result.fun**2)
    )

    return fitted_T, fitted_S, rmse


# ---------------------------------------------------------------------
# LOAD MODFLOW OBSERVATION DATA
# ---------------------------------------------------------------------

def load_h0_data():
    """
    Read the H0 MODFLOW observation workbook and select the
    fitting interval.

    Returns
    -------
    pandas.DataFrame
        MODFLOW observation data restricted to 0 < time <= 1 day.
    """

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
    )

    # Theis analysis requires time > 0.
    df = df[
        (df["time"] > 0.0)
        & (df["time"] <= FIT_END_TIME)
    ].copy()

    return df


# ---------------------------------------------------------------------
# SINGLE EXAMPLE FIT
# ---------------------------------------------------------------------

def run_example_fit(df):
    """
    Demonstrate Direct Theis parameter recovery using the
    50 m observation in Layer 1.
    """

    distance = 50.0

    time = df["time"].to_numpy()

    head = df[
        "HD_OBS_50M_1_101_103"
    ].to_numpy()

    # MODFLOW hydraulic head -> drawdown
    observed_drawdown = INITIAL_HEAD - head

    fitted_T, fitted_S, rmse = fit_theis_parameters(
        time=time,
        observed_drawdown=observed_drawdown,
        distance=distance,
    )

    print("\n--- DIRECT THEIS EXAMPLE: 50 m, LAYER 1 ---")

    print(
        f"Number of fitting points: "
        f"{len(time)}"
    )

    print(
        f"Reference T: "
        f"{T_TRUE:.3f} m2/day"
    )

    print(
        f"Fitted T: "
        f"{fitted_T:.3f} m2/day"
    )

    print(
        f"Reference S: "
        f"{S_TRUE:.4e}"
    )

    print(
        f"Fitted S: "
        f"{fitted_S:.4e}"
    )

    print(
        f"RMSE: "
        f"{rmse:.6e} m"
    )


# ---------------------------------------------------------------------
# ALL DISTANCES AND LAYERS
# ---------------------------------------------------------------------

def fit_all_observations(df):
    """
    Fit Direct Theis T and S estimates for all five observation
    distances and all three model layers.

    The workbook structure used in the dissertation contains the
    time column followed by three observation columns per distance.

    Returns
    -------
    pandas.DataFrame
        Parameter-recovery results for all observations.
    """

    time = df["time"].to_numpy()

    results = []

    required_observation_columns = (
        len(DISTANCES) * NUMBER_OF_LAYERS
    )

    available_observation_columns = (
        len(df.columns) - 1
    )

    if available_observation_columns < required_observation_columns:
        raise ValueError(
            "The observation workbook does not contain the expected "
            "five distances x three layers."
        )

    for distance_index, distance in enumerate(DISTANCES):

        for layer_index in range(NUMBER_OF_LAYERS):

            # Workbook organisation:
            # time column + 3 layers per observation distance
            column_number = (
                1
                + distance_index * NUMBER_OF_LAYERS
                + layer_index
            )

            column_name = df.columns[column_number]

            head_values = df[
                column_name
            ].to_numpy()

            observed_drawdown = (
                INITIAL_HEAD - head_values
            )

            fitted_T, fitted_S, rmse = (
                fit_theis_parameters(
                    time=time,
                    observed_drawdown=observed_drawdown,
                    distance=distance,
                )
            )

            T_bias = (
                (fitted_T - T_TRUE)
                / T_TRUE
                * 100.0
            )

            S_bias = (
                (fitted_S - S_TRUE)
                / S_TRUE
                * 100.0
            )

            results.append(
                {
                    "Distance_m": distance,
                    "Layer": layer_index + 1,
                    "Observation_column": column_name,
                    "Fitted_T_m2_day": fitted_T,
                    "T_Bias_percent": T_bias,
                    "Fitted_S": fitted_S,
                    "S_Bias_percent": S_bias,
                    "RMSE_m": rmse,
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------------------

def main():
    """Run H0 Direct Theis parameter recovery."""

    print(
        "Direct Theis Parameter Recovery "
        "from MODFLOW Output"
    )

    print(
        "---------------------------------------------"
    )

    print(
        f"Reference transmissivity: "
        f"{T_TRUE} m2/day"
    )

    print(
        f"Reference storativity: "
        f"{S_TRUE:.4e}"
    )

    print(
        f"Fitting window: "
        f"0 < t <= {FIT_END_TIME} day"
    )

    # Read MODFLOW observation data
    df = load_h0_data()

    # Demonstration fit at 50 m, Layer 1
    run_example_fit(df)

    # Fit all distances and layers
    results_df = fit_all_observations(df)

    print(
        "\n--- H0 ALL DIRECT THEIS FITS ---"
    )

    print(
        results_df.to_string(index=False)
    )

    # Save parameter-recovery results
    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
