"""
Transmissivity and Storativity Bias Analysis
============================================

This script combines Direct Theis and Cooper-Jacob parameter-recovery
results for the H0, H1 and H2 aquifer scenarios.

It calculates percentage bias in estimated transmissivity (T) and
storativity (S) relative to the prescribed synthetic aquifer values.

Scenarios
---------
H0 - Homogeneous control
H1 - Moderate vertical heterogeneity
H2 - Strong vertical heterogeneity

Methods
-------
Direct Theis
Cooper-Jacob

MSc project:
"Pumping-Test Simulation in MODFLOW"

Author: Rajalekshmi K Raju
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# REFERENCE PARAMETERS
# ---------------------------------------------------------------------

T_TRUE = 172.8          # Reference transmissivity [m2/day]
S_TRUE = 2.0e-4         # Reference storativity [-]


# ---------------------------------------------------------------------
# INPUT FILES
# ---------------------------------------------------------------------

DIRECT_FILES = {
    "H0": Path("H0_Direct_Theis_Fit.csv"),
    "H1": Path("H1_Direct_Theis_Fit.csv"),
    "H2": Path("H2_Direct_Theis_Fit.csv"),
}

CJ_FILES = {
    "H0": Path("H0_Cooper_Jacob_Fit.csv"),
    "H1": Path("H1_Cooper_Jacob_Fit.csv"),
    "H2": Path("H2_Cooper_Jacob_Fit.csv"),
}


# ---------------------------------------------------------------------
# OUTPUT FILES
# ---------------------------------------------------------------------

DETAILED_OUTPUT = Path("TS_All_Estimates_Bias.csv")
SUMMARY_OUTPUT = Path("TS_Bias_Summary.csv")


# ---------------------------------------------------------------------
# LOAD DIRECT THEIS RESULTS
# ---------------------------------------------------------------------

def load_direct_theis_results():
    """
    Load and standardise Direct Theis parameter estimates
    for H0, H1 and H2.
    """

    results = []

    for scenario, filename in DIRECT_FILES.items():

        df = pd.read_csv(filename)

        df = df.rename(
            columns={
                "Fitted_T_m2_day": "Estimated_T",
                "Fitted_S": "Estimated_S",
            }
        )

        df["Scenario"] = scenario
        df["Method"] = "Direct Theis"

        results.append(
            df[
                [
                    "Scenario",
                    "Method",
                    "Distance_m",
                    "Layer",
                    "Estimated_T",
                    "Estimated_S",
                    "RMSE_m",
                ]
            ]
        )

    return results


# ---------------------------------------------------------------------
# LOAD COOPER-JACOB RESULTS
# ---------------------------------------------------------------------

def load_cooper_jacob_results():
    """
    Load and standardise Cooper-Jacob parameter estimates
    for H0, H1 and H2.
    """

    results = []

    for scenario, filename in CJ_FILES.items():

        df = pd.read_csv(filename)

        df = df.rename(
            columns={
                "CJ_T_m2_day": "Estimated_T",
                "CJ_S": "Estimated_S",
            }
        )

        df["Scenario"] = scenario
        df["Method"] = "Cooper-Jacob"

        results.append(
            df[
                [
                    "Scenario",
                    "Method",
                    "Distance_m",
                    "Layer",
                    "Estimated_T",
                    "Estimated_S",
                    "RMSE_m",
                ]
            ]
        )

    return results


# ---------------------------------------------------------------------
# PARAMETER BIAS
# ---------------------------------------------------------------------

def calculate_parameter_bias(results):
    """
    Calculate percentage bias relative to the known
    aquifer-scale reference parameters.

    Positive bias indicates overestimation.
    Negative bias indicates underestimation.
    """

    results = results.copy()

    results["T_Bias_percent"] = (
        (results["Estimated_T"] - T_TRUE)
        / T_TRUE
        * 100.0
    )

    results["S_Bias_percent"] = (
        (results["Estimated_S"] - S_TRUE)
        / S_TRUE
        * 100.0
    )

    return results


# ---------------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------------

def create_summary(results):
    """
    Calculate mean recovered T, S and parameter bias
    for each scenario and analytical method.

    Note:
    RMSE is retained in the detailed results but is not used here
    for cross-method comparison because Direct Theis and Cooper-Jacob
    may use different valid observation subsets.

    Fair RMSE comparison is performed separately in
    fair_rmse_comparison.py.
    """

    summary = (
        results
        .groupby(
            ["Scenario", "Method"],
            as_index=False,
        )
        .agg(
            Mean_T=(
                "Estimated_T",
                "mean",
            ),
            Mean_S=(
                "Estimated_S",
                "mean",
            ),
            Mean_T_Bias_percent=(
                "T_Bias_percent",
                "mean",
            ),
            Mean_S_Bias_percent=(
                "S_Bias_percent",
                "mean",
            ),
            Valid_Estimates=(
                "Estimated_T",
                "count",
            ),
        )
    )

    return summary


# ---------------------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------------------

def main():
    """
    Combine parameter estimates and generate bias summaries.
    """

    direct_results = load_direct_theis_results()
    cj_results = load_cooper_jacob_results()

    # Combine all scenario and method results
    results = pd.concat(
        direct_results + cj_results,
        ignore_index=True,
    )

    # Calculate T and S bias
    results = calculate_parameter_bias(
        results
    )

    # Save detailed table
    results.to_csv(
        DETAILED_OUTPUT,
        index=False,
    )

    # Generate summary
    summary = create_summary(
        results
    )

    print(
        "\n--- TRANSMISSIVITY AND "
        "STORATIVITY BIAS SUMMARY ---"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # Save summary
    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print(
        f"\nDetailed results saved to: "
        f"{DETAILED_OUTPUT}"
    )

    print(
        f"Bias summary saved to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()
