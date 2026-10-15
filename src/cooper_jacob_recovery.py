"""
Cooper-Jacob Parameter Recovery from MODFLOW 6 Output
=====================================================

This script estimates transmissivity (T) and storativity (S) from
MODFLOW-generated hydraulic heads using the Cooper-Jacob
straight-line method.

Three controlled aquifer scenarios are analysed:

H0 - Homogeneous control
H1 - Moderate vertical heterogeneity
H2 - Strong vertical heterogeneity

Workflow
--------
1. Read MODFLOW observation heads.
2. Convert hydraulic head to drawdown.
3. Calculate the reference Theis parameter u.
4. Retain Cooper-Jacob-valid observations where u < 0.01.
5. Restrict the fitting interval to t <= 1 day.
6. Fit drawdown against log10(time).
7. Estimate transmissivity from the semi-log slope.
8. Estimate storativity from the zero-drawdown time intercept.
9. Calculate RMSE and parameter bias.
10. Repeat across five observation distances and three layers.

MSc project:
"Pumping-Test Simulation in MODFLOW"

Author: Rajalekshmi K Raju
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# REFERENCE MODEL PARAMETERS
# ---------------------------------------------------------------------

Q = 50.0               # Pumping-rate magnitude [m3/day]
INITIAL_HEAD = 10.0     # Initial hydraulic head [m]

T_TRUE = 172.8          # Reference transmissivity [m2/day]
S_TRUE = 2.0e-4         # Reference storativity [-]

DISTANCES = [50, 100, 200, 500, 1000]
NUMBER_OF_LAYERS = 3

FIT_END_TIME = 1.0      # Maximum Cooper-Jacob fitting time [day]
U_LIMIT = 0.01          # Cooper-Jacob validity criterion


# ---------------------------------------------------------------------
# SCENARIO CONFIGURATION
# ---------------------------------------------------------------------

SCENARIOS = {
    "H0": {
        "name": "Homogeneous control",
        "file": Path("3L_H0_Control_Validation.xlsx"),
        "file_type": "excel",
        "sheet": "H0_3L_Homo_Control.ob_gw",
        "output": Path("H0_Cooper_Jacob_Fit.csv"),
    },

    "H1": {
        "name": "Moderate vertical heterogeneity",
        "file": Path(
            "H1_Moderate_Heterogeneity.ob_gw_out_head.csv"
        ),
        "file_type": "csv",
        "sheet": None,
        "output": Path("H1_Cooper_Jacob_Fit.csv"),
    },

    "H2": {
        "name": "Strong vertical heterogeneity",
        "file": Path(
            "H2_Strong_Heterogeneity.ob_gw_out_head.csv"
        ),
        "file_type": "csv",
        "sheet": None,
        "output": Path("H2_Cooper_Jacob_Fit.csv"),
    },
}


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_scenario_data(config):
    """
    Read one MODFLOW observation dataset.

    H0 is stored in Excel.
    H1 and H2 are stored as CSV files.

    Returns
    -------
    pandas.DataFrame
        MODFLOW observation data.
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

    return df


# ---------------------------------------------------------------------
# COOPER-JACOB FITTING
# ---------------------------------------------------------------------

def fit_cooper_jacob(
    time,
    drawdown,
    distance,
):
    """
    Estimate transmissivity and storativity using the
    Cooper-Jacob straight-line method.

    Only observations satisfying:

        u < 0.01

    and:

        0 < t <= 1 day

    are used.

    Parameters
    ----------
    time : numpy.ndarray
        Simulation time [day].
    drawdown : numpy.ndarray
        MODFLOW-derived drawdown [m].
    distance : float
        Observation distance from pumping well [m].

    Returns
    -------
    dict
        Cooper-Jacob estimates and fitting statistics.
    """

    # Initialise u with infinity so t = 0 is automatically invalid
    u_reference = np.full(
        time.shape,
        np.inf,
        dtype=float,
    )

    positive_time = time > 0.0

    # Calculate reference Theis parameter only for positive times
    u_reference[positive_time] = (
        distance**2
        * S_TRUE
        / (
            4.0
            * T_TRUE
            * time[positive_time]
        )
    )

    # Cooper-Jacob validity screening
    valid = (
        positive_time
        & (u_reference < U_LIMIT)
        & (time <= FIT_END_TIME)
    )

    time_cj = time[valid]
    drawdown_cj = drawdown[valid]

    # At least three points are required for the regression
    if len(time_cj) < 3:

        return {
            "CJ_T_m2_day": np.nan,
            "T_Bias_percent": np.nan,
            "CJ_S": np.nan,
            "S_Bias_percent": np.nan,
            "Slope": np.nan,
            "Intercept": np.nan,
            "t0_day": np.nan,
            "Points_used": len(time_cj),
            "RMSE_m": np.nan,
        }

    # -------------------------------------------------------------
    # Straight-line regression:
    # drawdown versus log10(time)
    # -------------------------------------------------------------

    log_time = np.log10(
        time_cj
    )

    slope, intercept = np.polyfit(
        log_time,
        drawdown_cj,
        1,
    )

    # -------------------------------------------------------------
    # Transmissivity
    # -------------------------------------------------------------

    T_cj = (
        2.3 * Q
        / (4.0 * np.pi * slope)
    )

    # -------------------------------------------------------------
    # Zero-drawdown time intercept
    # -------------------------------------------------------------

    log_t0 = (
        -intercept
        / slope
    )

    t0 = 10 ** log_t0

    # -------------------------------------------------------------
    # Storativity
    # -------------------------------------------------------------

    S_cj = (
        2.25
        * T_cj
        * t0
        / distance**2
    )

    # -------------------------------------------------------------
    # Regression RMSE
    # -------------------------------------------------------------

    fitted_drawdown = (
        slope
        * log_time
        + intercept
    )

    rmse = np.sqrt(
        np.mean(
            (
                fitted_drawdown
                - drawdown_cj
            ) ** 2
        )
    )

    # -------------------------------------------------------------
    # Parameter bias
    # -------------------------------------------------------------

    T_bias = (
        (T_cj - T_TRUE)
        / T_TRUE
        * 100.0
    )

    S_bias = (
        (S_cj - S_TRUE)
        / S_TRUE
        * 100.0
    )

    return {
        "CJ_T_m2_day": T_cj,
        "T_Bias_percent": T_bias,
        "CJ_S": S_cj,
        "S_Bias_percent": S_bias,
        "Slope": slope,
        "Intercept": intercept,
        "t0_day": t0,
        "Points_used": len(time_cj),
        "RMSE_m": rmse,
    }


# ---------------------------------------------------------------------
# FIT ALL OBSERVATIONS
# ---------------------------------------------------------------------

def fit_all_observations(
    df,
    scenario,
):
    """
    Apply Cooper-Jacob analysis to all five observation
    distances and three aquifer layers.

    Returns
    -------
    pandas.DataFrame
        Cooper-Jacob parameter-recovery results.
    """

    time = df[
        "time"
    ].to_numpy()

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

            head = df[
                column_name
            ].to_numpy()

            # MODFLOW hydraulic head -> drawdown
            drawdown = (
                INITIAL_HEAD
                - head
            )

            fit_result = (
                fit_cooper_jacob(
                    time=time,
                    drawdown=drawdown,
                    distance=distance,
                )
            )

            results.append(
                {
                    "Scenario": scenario,
                    "Distance_m": distance,
                    "Layer": layer_index + 1,
                    "Observation_column": (
                        column_name
                    ),
                    **fit_result,
                }
            )

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------------------
# SCENARIO SUMMARY
# ---------------------------------------------------------------------

def summarise_results(
    scenario,
    scenario_name,
    results_df,
):
    """
    Print mean Cooper-Jacob parameter estimates for
    one heterogeneity scenario.
    """

    valid_results = (
        results_df.dropna(
            subset=[
                "CJ_T_m2_day",
                "CJ_S",
            ]
        )
    )

    print()
    print(
        f"--- {scenario}: "
        f"{scenario_name} ---"
    )

    print(
        f"Valid fits = "
        f"{len(valid_results)}"
    )

    if valid_results.empty:

        print(
            "No valid Cooper-Jacob fits."
        )

        return

    mean_T = (
        valid_results[
            "CJ_T_m2_day"
        ].mean()
    )

    mean_S = (
        valid_results[
            "CJ_S"
        ].mean()
    )

    mean_T_bias = (
        valid_results[
            "T_Bias_percent"
        ].mean()
    )

    mean_S_bias = (
        valid_results[
            "S_Bias_percent"
        ].mean()
    )

    mean_rmse = (
        valid_results[
            "RMSE_m"
        ].mean()
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
    Run Cooper-Jacob parameter recovery for one scenario.
    """

    print()
    print(
        "======================================"
    )

    print(
        f"{scenario}: "
        f"{config['name']}"
    )

    print(
        "======================================"
    )

    df = load_scenario_data(
        config
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
    Run Cooper-Jacob parameter recovery for H0, H1 and H2.
    """

    print(
        "MODFLOW Cooper-Jacob "
        "Parameter Recovery"
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
        f"Validity criterion: "
        f"u < {U_LIMIT}"
    )

    print(
        f"Fitting window: "
        f"0 < t <= {FIT_END_TIME} day"
    )

    all_results = []

    for scenario, config in (
        SCENARIOS.items()
    ):

        results_df = (
            run_scenario(
                scenario=scenario,
                config=config,
            )
        )

        all_results.append(
            results_df
        )

    combined_results = (
        pd.concat(
            all_results,
            ignore_index=True,
        )
    )

    combined_results.to_csv(
        "Cooper_Jacob_All_Scenarios.csv",
        index=False,
    )

    print()

    print(
        "Combined results saved to: "
        "Cooper_Jacob_All_Scenarios.csv"
    )


if __name__ == "__main__":
    main()
