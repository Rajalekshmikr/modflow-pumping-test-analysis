"""
Fair RMSE Comparison: Direct Theis vs Cooper-Jacob
==================================================

This script compares Direct Theis and Cooper-Jacob fitting RMSE
using the same valid observation subset.

The purpose is to ensure that the two methods are compared fairly:
Direct Theis is re-fitted only on the observations that satisfy the
Cooper-Jacob validity criterion (u < 0.01) within the selected
time window.

Scenarios analysed:
H0 - Homogeneous control
H1 - Moderate vertical heterogeneity
H2 - Strong vertical heterogeneity

MSc project:
"Pumping-Test Simulation in MODFLOW"

Author: Rajalekshmi K Raju
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import exp1


# ---------------------------------------------------------------------
# REFERENCE PARAMETERS
# ---------------------------------------------------------------------

Q = 50.0
INITIAL_HEAD = 10.0
T_TRUE = 172.8
S_TRUE = 2.0e-4

FIT_END_TIME = 1.0
U_LIMIT = 0.01

# In this comparison script, only the first two observation distances
# were included in the original workflow.
DISTANCES = [50, 100]
LAYERS = [1, 2, 3]


# ---------------------------------------------------------------------
# FILE CONFIGURATION
# ---------------------------------------------------------------------

OBSERVATION_FILES = {
    "H0": {
        "file_type": "excel",
        "file": Path("3L_H0_Control_Validation.xlsx"),
        "sheet": "H0_3L_Homo_Control.ob_gw",
    },
    "H1": {
        "file_type": "csv",
        "file": Path("H1_Moderate_Heterogeneity.ob_gw_out_head.csv"),
        "sheet": None,
    },
    "H2": {
        "file_type": "csv",
        "file": Path("H2_Strong_Heterogeneity.ob_gw_out_head.csv"),
        "sheet": None,
    },
}

CJ_FILES = {
    "H0": Path("H0_Cooper_Jacob_Fit.csv"),
    "H1": Path("H1_Cooper_Jacob_Fit.csv"),
    "H2": Path("H2_Cooper_Jacob_Fit.csv"),
}

OUTPUT_TABLE = Path("Fair_RMSE_Comparison.csv")
OUTPUT_FIGURE = Path("Fair_RMSE_Comparison.png")


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def load_observation_data(config):
    """Load one observation dataset."""

    if config["file_type"] == "excel":
        df = pd.read_excel(
            config["file"],
            sheet_name=config["sheet"],
        )
    elif config["file_type"] == "csv":
        df = pd.read_csv(config["file"])
    else:
        raise ValueError("Unsupported observation file type.")

    return df


def theis_drawdown(time, distance, transmissivity, storativity):
    """Calculate Theis drawdown."""

    u = (distance**2 * storativity) / (4.0 * transmissivity * time)
    return (Q / (4.0 * np.pi * transmissivity)) * exp1(u)


def valid_cj_window(time, distance):
    """
    Return a boolean mask for observations satisfying:
    - positive time
    - u < 0.01
    - time <= 1 day
    """

    positive_time = time > 0.0

    u_reference = np.full_like(time, np.inf, dtype=float)
    u_reference[positive_time] = (
        distance**2 * S_TRUE / (4.0 * T_TRUE * time[positive_time])
    )

    valid = (
        positive_time
        & (u_reference < U_LIMIT)
        & (time <= FIT_END_TIME)
    )

    return valid


def direct_theis_rmse(time_valid, observed_valid, distance):
    """
    Fit Direct Theis using the same valid points retained for
    Cooper-Jacob, then calculate RMSE.
    """

    if len(time_valid) < 3:
        return np.nan

    def residuals(log_parameters):
        transmissivity = 10 ** log_parameters[0]
        storativity = 10 ** log_parameters[1]

        predicted = theis_drawdown(
            time=time_valid,
            distance=distance,
            transmissivity=transmissivity,
            storativity=storativity,
        )

        return predicted - observed_valid

    fit = least_squares(
        residuals,
        [np.log10(T_TRUE), np.log10(S_TRUE)],
    )

    rmse = np.sqrt(np.mean(fit.fun**2))
    return rmse


# ---------------------------------------------------------------------
# MAIN COMPARISON
# ---------------------------------------------------------------------

def main():
    """Run fair RMSE comparison for H0, H1 and H2."""

    summary_rows = []

    for scenario in ["H0", "H1", "H2"]:

        obs = load_observation_data(
            OBSERVATION_FILES[scenario]
        )

        cj = pd.read_csv(CJ_FILES[scenario])

        direct_rmse_values = []
        cj_rmse_values = []

        for distance_index, distance in enumerate(DISTANCES):

            for layer in LAYERS:

                column_number = 1 + distance_index * 3 + (layer - 1)
                column_name = obs.columns[column_number]

                time = obs["time"].to_numpy()
                head = obs[column_name].to_numpy()
                observed_drawdown = INITIAL_HEAD - head

                # SAME Cooper-Jacob validity window for both methods
                valid = valid_cj_window(
                    time=time,
                    distance=distance,
                )

                time_valid = time[valid]
                observed_valid = observed_drawdown[valid]

                # Direct Theis RMSE using same valid points
                direct_rmse = direct_theis_rmse(
                    time_valid=time_valid,
                    observed_valid=observed_valid,
                    distance=distance,
                )

                direct_rmse_values.append(direct_rmse)

                # Cooper-Jacob RMSE from previously exported results
                cj_row = cj[
                    (cj["Distance_m"] == distance)
                    & (cj["Layer"] == layer)
                ].iloc[0]

                cj_rmse_values.append(cj_row["RMSE_m"])

        summary_rows.append(
            {
                "Scenario": scenario,
                "Direct_Theis_RMSE": np.nanmean(direct_rmse_values),
                "Cooper_Jacob_RMSE": np.nanmean(cj_rmse_values),
            }
        )

    summary = pd.DataFrame(summary_rows)

    print("\n--- FAIR RMSE COMPARISON ---")
    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_TABLE,
        index=False,
    )

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------

    x = np.arange(len(summary))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        summary["Direct_Theis_RMSE"],
        width,
        label="Direct Theis",
    )

    plt.bar(
        x + width / 2,
        summary["Cooper_Jacob_RMSE"],
        width,
        label="Cooper-Jacob",
    )

    plt.xticks(x, summary["Scenario"])
    plt.xlabel("Heterogeneity Scenario")
    plt.ylabel("RMSE (m)")
    plt.title("RMSE Comparison Using Common Valid Observation Data")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE,
        dpi=300,
    )

    plt.show()

    print(f"\nSummary table saved to: {OUTPUT_TABLE}")
    print(f"Figure saved to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
