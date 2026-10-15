"""
Parameter Bias Plots
====================

This script visualises the mean transmissivity (T) bias and
mean storativity (S) bias for Direct Theis and Cooper-Jacob
across the H0, H1 and H2 scenarios.

It also optionally plots fair RMSE comparison results from
Fair_RMSE_Comparison.csv, which compares the two methods
using the same valid observation subset.

MSc project:
"Pumping-Test Simulation in MODFLOW"

Author: Rajalekshmi K Raju
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# INPUT FILES
# ---------------------------------------------------------------------

BIAS_SUMMARY_FILE = Path("TS_Bias_Summary.csv")
FAIR_RMSE_FILE = Path("Fair_RMSE_Comparison.csv")

SCENARIOS = ["H0", "H1", "H2"]
METHODS = ["Direct Theis", "Cooper-Jacob"]


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

df = pd.read_csv(BIAS_SUMMARY_FILE)


# ---------------------------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------------------------

def extract_values(dataframe, value_column):
    """
    Extract scenario-wise values for Direct Theis and Cooper-Jacob.
    """

    theis_values = []
    cj_values = []

    for scenario in SCENARIOS:

        theis_values.append(
            dataframe[
                (dataframe["Scenario"] == scenario)
                & (dataframe["Method"] == "Direct Theis")
            ][value_column].values[0]
        )

        cj_values.append(
            dataframe[
                (dataframe["Scenario"] == scenario)
                & (dataframe["Method"] == "Cooper-Jacob")
            ][value_column].values[0]
        )

    return theis_values, cj_values


# ---------------------------------------------------------------------
# GRAPH 1: MEAN T BIAS
# ---------------------------------------------------------------------

x = np.arange(len(SCENARIOS))
width = 0.35

theis_t, cj_t = extract_values(
    df,
    "Mean_T_Bias_percent",
)

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    theis_t,
    width,
    label="Direct Theis",
)

plt.bar(
    x + width / 2,
    cj_t,
    width,
    label="Cooper-Jacob",
)

plt.xticks(x, SCENARIOS)
plt.xlabel("Scenario")
plt.ylabel("Mean T Bias (%)")
plt.title("Comparison of Mean Transmissivity Bias Across Scenarios")
plt.legend()
plt.tight_layout()

plt.savefig(
    "Mean_T_Bias_Comparison.png",
    dpi=300,
)

plt.show()


# ---------------------------------------------------------------------
# GRAPH 2: MEAN S BIAS
# ---------------------------------------------------------------------

theis_s, cj_s = extract_values(
    df,
    "Mean_S_Bias_percent",
)

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    theis_s,
    width,
    label="Direct Theis",
)

plt.bar(
    x + width / 2,
    cj_s,
    width,
    label="Cooper-Jacob",
)

plt.xticks(x, SCENARIOS)
plt.xlabel("Scenario")
plt.ylabel("Mean S Bias (%)")
plt.title("Comparison of Mean Storativity Bias Across Scenarios")
plt.legend()
plt.tight_layout()

plt.savefig(
    "Mean_S_Bias_Comparison.png",
    dpi=300,
)

plt.show()


# ---------------------------------------------------------------------
# OPTIONAL GRAPH 3: FAIR RMSE COMPARISON
# ---------------------------------------------------------------------

if FAIR_RMSE_FILE.exists():

    rmse_df = pd.read_csv(FAIR_RMSE_FILE)

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        rmse_df["Direct_Theis_RMSE"],
        width,
        label="Direct Theis",
    )

    plt.bar(
        x + width / 2,
        rmse_df["Cooper_Jacob_RMSE"],
        width,
        label="Cooper-Jacob",
    )

    plt.xticks(x, rmse_df["Scenario"])
    plt.xlabel("Scenario")
    plt.ylabel("RMSE (m)")
    plt.title("Fair RMSE Comparison Using Common Valid Observation Data")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "Fair_RMSE_Comparison.png",
        dpi=300,
    )

    plt.show()

    print("\nFair RMSE plot created from Fair_RMSE_Comparison.csv")

else:
    print(
        "\nFair_RMSE_Comparison.csv not found. "
        "Skipping RMSE plot."
    )
