"""
Direct Theis Parameter Recovery from MODFLOW 6 Output
=====================================================

This script estimates transmissivity (T) and storativity (S) from
MODFLOW-generated hydraulic heads using nonlinear least-squares
fitting of the full Theis solution.

Three controlled aquifer scenarios are analysed:

H0 - Homogeneous control
H1 - Moderate vertical heterogeneity
H2 - Strong vertical heterogeneity

Workflow
--------
1. Read MODFLOW observation heads.
2. Convert hydraulic head to drawdown.
3. Restrict analysis to 0 < t <= 1 day.
4. Fit transmissivity and storativity using Direct Theis fitting.
5. Calculate RMSE and parameter bias.
6. Repeat across five observation distances and three model layers.
7. Export scenario-specific parameter-recovery tables.

MSc project:
"Pumping-Test Simulation in MODFLOW"

Author: Rajalekshmi K Raju
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import exp1


# ---------------------------------------------------------------------
# REFERENCE MODEL PARAMETERS
# ---------------------------------------------------------------------

Q = 50.0               # Pumping-rate magnitude [m3/day]
INITIAL_HEAD = 10.0     # Initial hydraulic head [m]

T_TRUE = 172.8          # Reference transmissivity [m2/day]
S_TRUE = 2.0e-4         # Reference storativity [-]

DISTANCES = [50, 100, 200, 500, 1000]
NUMBER_OF_LAYERS = 3

FIT_END_TIME = 1.0      # Direct Theis fitting window [day]


# ---------------------------------------------------------------------
# SCENARIO CONFIGURATION
# ---------------------------------------------------------------------

SCENARIOS = {
    "H0": {
        "name": "Homogeneous control",
        "file": Path("3L_H0_Control_Validation.xlsx"),
        "file_type": "excel",
        "sheet": "H0_3L_Homo_Control.ob_gw",
        "output": Path("H0_Direct_Theis_Fit.csv"),
    },

    "H1": {
        "name": "Moderate vertical heterogeneity",
        "file": Path(
            "H1_Moderate_Heterogeneity.ob_gw_out_head.csv"
        ),
        "file_type": "csv",
        "sheet": None,
        "output": Path("H1_Direct_Theis_Fit.csv"),
    },

    "H2": {
        "name": "Strong vertical heterogeneity",
        "file": Path(
            "H2_Strong_Heterogeneity.ob_gw_out_head.csv"
        ),
        "file_type": "csv",
        "sheet": None,
        "output": Path("H2_Direct_Theis_Fit.csv"),
    },
}


# ---------------------------------------------------------------------
# THEIS SOLUTION
# ---------------------------------------------------------------------

def theis_drawdown(
    time,
    distance,
    transmissivity,
    storativity,
):
    """
    Calculate analytical Theis drawdown.

    Parameters
    ----------
    time : numpy.ndarray
        Time since pumping started [day].
    distance : float
        Radial distance from pumping well [m].
    transmissivity : float
        Transmissivity [m2/day].
    storativity : float
        Storativity [-].

    Returns
    -------
    numpy.ndarray
        Analytical drawdown [m].
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
# DIRECT THEIS PARAMETER FITTING
# ---------------------------------------------------------------------

def fit_theis_parameters(
    time,
    observed_drawdown,
    distance,
):
    """
    Estimate T and S using nonlinear least-squares fitting
    of the full Theis solution.

    Optimisation is performed in log10 parameter space,
    keeping fitted T and S positive.

    Returns
    -------
    fitted_T : float
        Estimated transmissivity [m2/day].
    fitted_S : float
        Estimated storativity [-].
    rmse : float
        Drawdown fitting RMSE [m].
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

        return (
            predicted_drawdown
            - observed_drawdown
        )

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
# DATA LOADING
# ---------------------------------------------------------------------

def load_scenario_data(config):
    """
    Read one MODFLOW observation dataset.

    H0 is stored in an Excel workbook.
    H1 and H2 are stored as CSV files.
    """

    if config["file_type"] == "excel":

        df = pd.read_excel(
            config["file"],
            sheet_name=config["sheet"],
        )

    elif config["file_type"] == "csv":

        df = pd.read_csv(
            config["file"]
        )

    else:
        raise ValueError(
            "Unsupported input file type."
        )

    # Exclude t = 0 because the Theis equation contains 1/t.
    # Restrict fitting to the selected early-time window.
    df = df[
        (df["time"] > 0.0)
        & (df["time"] <= FIT_END_TIME)
    ].copy()

    return df


# ---------------------------------------------------------------------
# FIT ALL OBSERVATIONS
# ---------------------------------------------------------------------

def fit_all_observations(
    df,
    scenario,
):
    """
    Estimate T and S for all five observation distances
    and all three model layers.

    The dissertation observation tables contain the time
    column followed by three layer observations for each
    distance.

    Returns
    -------
    pandas.DataFrame
        Parameter-recovery results.
    """

    time = df["time"].to_numpy()

    results = []

    expected_observation_columns = (
        len(DISTANCES)
        * NUMBER_OF_LAYERS
    )

    available_observation_columns = (
        len(df.columns) - 1
    )

    if (
        available_observation_columns
        < expected_observation_columns
    ):
        raise ValueError(
            f"{scenario}: expected at least "
            f"{expected_observation_columns} "
            "observation columns."
        )

    for distance_index, distance in enumerate(
        DISTANCES
    ):

        for layer_index in range(
            NUMBER_OF_LAYERS
        ):

            column_number = (
                1
                + distance_index
                * NUMBER_OF_LAYERS
                + layer_index
            )

            column_name = (
                df.columns[column_number]
            )

            head_values = (
                df[column_name]
                .to_numpy()
            )

            # MODFLOW hydraulic head -> drawdown
            observed_drawdown = (
                INITIAL_HEAD
                - head_values
            )

            fitted_T, fitted_S, rmse = (
                fit_theis_parameters(
                    time=time,
                    observed_drawdown=(
                        observed_drawdown
                    ),
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
                    "Scenario": scenario,
                    "Distance_m": distance,
                    "Layer": layer_index + 1,
                    "Observation_column": (
                        column_name
                    ),
                    "Fitted_T_m2_day": (
                        fitted_T
                    ),
                    "T_Bias_percent": (
                        T_bias
                    ),
                    "Fitted_S": fitted_S,
                    "S_Bias_percent": (
                        S_bias
                    ),
                    "RMSE_m": rmse,
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# SCENARIO SUMMARY
# ---------------------------------------------------------------------

def summarise_results(
    scenario,
    scenario_name,
    results_df,
):
    """
    Print mean parameter estimates and bias for one scenario.
    """

    mean_T = (
        results_df["Fitted_T_m2_day"]
        .mean()
    )

    mean_S = (
        results_df["Fitted_S"]
        .mean()
    )

    mean_T_bias = (
        results_df["T_Bias_percent"]
        .mean()
    )

    mean_S_bias = (
        results_df["S_Bias_percent"]
        .mean()
    )

    mean_rmse = (
        results_df["RMSE_m"]
        .mean()
    )

    print()
    print(
        f"--- {scenario}: "
        f"{scenario_name} ---"
    )

    print(
        f"Mean fitted T = "
        f"{mean_T:.3f} m2/day"
    )

    print(
        f"Mean T bias = "
        f"{mean_T_bias:.3f}%"
    )

    print(
        f"Mean fitted S = "
        f"{mean_S:.4e}"
    )

    print(
        f"Mean S bias = "
        f"{mean_S_bias:.3f}%"
    )

    print(
        f"Mean RMSE = "
        f"{mean_rmse:.6e} m"
    )


# ---------------------------------------------------------------------
# RUN ONE SCENARIO
# ---------------------------------------------------------------------

def run_scenario(
    scenario,
    config,
):
    """
    Execute Direct Theis parameter recovery
    for one MODFLOW scenario.
    """

    print()
    print(
        "======================================"
    )

    print(
        f"{scenario}: {config['name']}"
    )

    print(
        "======================================"
    )

    df = load_scenario_data(
        config
    )

    print(
        f"Number of fitting times: "
        f"{len(df)}"
    )

    results_df = (
        fit_all_observations(
            df=df,
            scenario=scenario,
        )
    )

    summarise_results(
        scenario=scenario,
        scenario_name=config["name"],
        results_df=results_df,
    )

    results_df.to_csv(
        config["output"],
        index=False,
    )

    print(
        f"Results saved to: "
        f"{config['output']}"
    )

    return results_df


# ---------------------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------------------

def main():
    """
    Run Direct Theis parameter recovery for H0, H1 and H2.
    """

    print(
        "MODFLOW Direct Theis Parameter Recovery"
    )

    print(
        "======================================="
    )

    print(
        f"Reference T = "
        f"{T_TRUE:.3f} m2/day"
    )

    print(
        f"Reference S = "
        f"{S_TRUE:.4e}"
    )

    print(
        f"Fitting window = "
        f"0 < t <= {FIT_END_TIME} day"
    )

    all_results = []

    for scenario, config in (
        SCENARIOS.items()
    ):

        results_df = run_scenario(
            scenario=scenario,
            config=config,
        )

        all_results.append(
            results_df
        )

    combined_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_results.to_csv(
        "Direct_Theis_All_Scenarios.csv",
        index=False,
    )

    print()
    print(
        "Combined results saved to: "
        "Direct_Theis_All_Scenarios.csv"
    )


if __name__ == "__main__":
    main()
