"""
Analytical Pumping-Test Benchmark
=================================

This script evaluates transient drawdown in a synthetic confined aquifer using:

1. Theis analytical solution
2. Cooper-Jacob late-time approximation
3. Relative-error comparison between the two methods
4. Cooper-Jacob estimation of transmissivity and storativity

The script forms part of the MSc project:
"Pumping-Test Simulation in MODFLOW"

Author: Rajalekshmi K Raju
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import exp1


# ---------------------------------------------------------------------
# AQUFER AND PUMPING PARAMETERS
# ---------------------------------------------------------------------

K = 8.64        # Hydraulic conductivity [m/day]
b = 20.0        # Aquifer thickness [m]
Ss = 1.0e-5     # Specific storage [1/m]
Q = 50.0        # Pumping-rate magnitude [m3/day]
h0 = 10.0       # Initial hydraulic head [m]

# Derived aquifer properties
T = K * b       # Transmissivity [m2/day]
S = Ss * b      # Storativity [-]

# Simulation times used in the analytical comparison
TIMES = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0]

# Analytical calculations were evaluated at additional radii.
# The principal dissertation observation distances were
# 50, 100, 200, 500 and 1000 m.
DISTANCES = list(range(50, 501, 50)) + [1000]

PRINCIPAL_DISTANCES = [50, 100, 200, 500, 1000]


# ---------------------------------------------------------------------
# ANALYTICAL FUNCTIONS
# ---------------------------------------------------------------------

def calculate_theis(Q, T, S, r, t, h0):
    """
    Calculate Theis drawdown and final hydraulic head.

    Parameters
    ----------
    Q : float
        Pumping-rate magnitude [m3/day].
    T : float
        Transmissivity [m2/day].
    S : float
        Storativity [-].
    r : float
        Radial distance from the pumping well [m].
    t : float
        Time since pumping began [day].
    h0 : float
        Initial hydraulic head [m].

    Returns
    -------
    u : float
        Dimensionless Theis parameter.
    well_function : float
        Theis well function W(u).
    drawdown : float
        Theis drawdown [m].
    final_head : float
        Hydraulic head after drawdown [m].
    """

    if r <= 0:
        raise ValueError("Distance r must be greater than zero.")

    if t <= 0:
        raise ValueError("Time t must be greater than zero.")

    if T <= 0:
        raise ValueError("Transmissivity T must be greater than zero.")

    if S <= 0:
        raise ValueError("Storativity S must be greater than zero.")

    # Theis dimensionless parameter
    u = (r**2 * S) / (4.0 * T * t)

    # Exponential integral represents the Theis well function
    well_function = exp1(u)

    # Drawdown
    drawdown = Q * well_function / (4.0 * math.pi * T)

    # Final hydraulic head
    final_head = h0 - drawdown

    return u, well_function, drawdown, final_head


def calculate_cooper_jacob(Q, T, u, h0):
    """
    Calculate Cooper-Jacob approximate drawdown.

    The approximation is applied only where u < 0.01.

    Parameters
    ----------
    Q : float
        Pumping-rate magnitude [m3/day].
    T : float
        Transmissivity [m2/day].
    u : float
        Theis dimensionless parameter.
    h0 : float
        Initial hydraulic head [m].

    Returns
    -------
    well_function : float or None
        Cooper-Jacob approximation of the well function.
    drawdown : float or None
        Cooper-Jacob drawdown [m].
    final_head : float or None
        Hydraulic head after drawdown [m].
    """

    if u >= 0.01:
        return None, None, None

    # Late-time approximation to the Theis well function
    well_function = -0.5772 - math.log(u)

    drawdown = Q * well_function / (4.0 * math.pi * T)

    final_head = h0 - drawdown

    return well_function, drawdown, final_head


# ---------------------------------------------------------------------
# GENERATE ANALYTICAL RESULTS
# ---------------------------------------------------------------------

def generate_results():
    """
    Calculate Theis and Cooper-Jacob responses for all specified
    times and observation distances.

    Returns
    -------
    pandas.DataFrame
        Table containing analytical pumping-test results.
    """

    results = []

    for t in TIMES:
        for r in DISTANCES:

            (
                u,
                theis_well_function,
                theis_drawdown,
                theis_final_head,
            ) = calculate_theis(
                Q=Q,
                T=T,
                S=S,
                r=r,
                t=t,
                h0=h0,
            )

            (
                cj_well_function,
                cj_drawdown,
                cj_final_head,
            ) = calculate_cooper_jacob(
                Q=Q,
                T=T,
                u=u,
                h0=h0,
            )

            results.append(
                {
                    "Distance (m)": r,
                    "Time (day)": t,
                    "u": u,
                    "Theis well function": theis_well_function,
                    "Theis drawdown (m)": theis_drawdown,
                    "Theis final head (m)": theis_final_head,
                    "CJ valid": u < 0.01,
                    "CJ well function": cj_well_function,
                    "CJ drawdown (m)": cj_drawdown,
                    "CJ final head (m)": cj_final_head,
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# ERROR ANALYSIS
# ---------------------------------------------------------------------

def calculate_cj_error(df, distance=50):
    """
    Compare Cooper-Jacob drawdown with Theis drawdown where
    the Cooper-Jacob approximation is valid.

    Parameters
    ----------
    df : pandas.DataFrame
        Analytical results table.
    distance : float
        Observation distance [m].

    Returns
    -------
    pandas.DataFrame
        Valid observations with absolute and relative errors.
    """

    valid_data = df[
        (df["Distance (m)"] == distance)
        & (df["CJ valid"])
    ].copy()

    valid_data["Absolute difference (m)"] = np.abs(
        valid_data["CJ drawdown (m)"]
        - valid_data["Theis drawdown (m)"]
    )

    valid_data["Relative error (%)"] = (
        valid_data["Absolute difference (m)"]
        / valid_data["Theis drawdown (m)"]
    ) * 100.0

    return valid_data


# ---------------------------------------------------------------------
# COOPER-JACOB PARAMETER ESTIMATION
# ---------------------------------------------------------------------

def cooper_jacob_two_point_estimate(
    df,
    distance=50,
    t1=0.1,
    t2=1.0,
):
    """
    Estimate transmissivity and storativity using two points
    separated by one logarithmic time cycle.

    Parameters
    ----------
    df : pandas.DataFrame
        Valid Cooper-Jacob analytical data.
    distance : float
        Observation distance [m].
    t1 : float
        First time [day].
    t2 : float
        Second time [day].

    Returns
    -------
    T_est : float
        Estimated transmissivity [m2/day].
    t0 : float
        Zero-drawdown time intercept [day].
    S_est : float
        Estimated storativity [-].
    """

    data = df[df["Distance (m)"] == distance].copy()

    row_t1 = data[np.isclose(data["Time (day)"], t1)]
    row_t2 = data[np.isclose(data["Time (day)"], t2)]

    if row_t1.empty or row_t2.empty:
        raise ValueError(
            "Requested Cooper-Jacob fitting times were not found."
        )

    if not bool(row_t1["CJ valid"].iloc[0]):
        raise ValueError(f"Cooper-Jacob is not valid at t = {t1} day.")

    if not bool(row_t2["CJ valid"].iloc[0]):
        raise ValueError(f"Cooper-Jacob is not valid at t = {t2} day.")

    s1 = row_t1["CJ drawdown (m)"].iloc[0]
    s2 = row_t2["CJ drawdown (m)"].iloc[0]

    # Drawdown change across the selected logarithmic interval
    delta_s = s2 - s1

    # Cooper-Jacob transmissivity estimate
    T_est = 0.183 * Q / delta_s

    # Extrapolated zero-drawdown intercept
    t0 = t1 / (10 ** (s1 / delta_s))

    # Cooper-Jacob storativity estimate
    S_est = 2.25 * T_est * t0 / (distance**2)

    return T_est, t0, S_est


def cooper_jacob_regression(df, distance=50):
    """
    Estimate transmissivity and storativity from a straight-line
    regression through all valid Cooper-Jacob observations.

    Parameters
    ----------
    df : pandas.DataFrame
        Analytical pumping-test results.
    distance : float
        Observation distance [m].

    Returns
    -------
    slope : float
        Semi-log drawdown slope.
    intercept : float
        Regression intercept.
    T_est : float
        Estimated transmissivity [m2/day].
    t0 : float
        Zero-drawdown time intercept [day].
    S_est : float
        Estimated storativity [-].
    """

    data = df[
        (df["Distance (m)"] == distance)
        & (df["CJ valid"])
    ].copy()

    if len(data) < 2:
        raise ValueError(
            "At least two valid Cooper-Jacob observations are required."
        )

    log_time = np.log10(data["Time (day)"])
    drawdown = data["CJ drawdown (m)"]

    slope, intercept = np.polyfit(
        log_time,
        drawdown,
        1,
    )

    T_est = 0.183 * Q / slope

    t0 = 10 ** (-intercept / slope)

    S_est = 2.25 * T_est * t0 / (distance**2)

    return slope, intercept, T_est, t0, S_est


# ---------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------------------

def plot_theis_at_50m(df):
    """Plot Theis drawdown through time at 50 m."""

    data = df[df["Distance (m)"] == 50]

    plt.figure()

    plt.plot(
        data["Time (day)"],
        data["Theis drawdown (m)"],
        marker="o",
    )

    plt.xscale("log")
    plt.xlabel("Time (day)")
    plt.ylabel("Drawdown (m)")
    plt.title("Theis Drawdown at r = 50 m")
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_theis_all_distances(df):
    """
    Plot Theis drawdown at the five principal observation distances.
    """

    plt.figure()

    for r_plot in PRINCIPAL_DISTANCES:

        data = df[df["Distance (m)"] == r_plot]

        plt.plot(
            data["Time (day)"],
            data["Theis drawdown (m)"],
            marker="o",
            label=f"{r_plot} m",
        )

    plt.xscale("log")
    plt.xlabel("Time (day)")
    plt.ylabel("Drawdown (m)")
    plt.title("Theis Drawdown at Different Observation Distances")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_theis_vs_cj(df, distance=50):
    """
    Compare Theis and Cooper-Jacob drawdown where the
    Cooper-Jacob approximation is valid.
    """

    data = df[
        (df["Distance (m)"] == distance)
        & (df["CJ valid"])
    ]

    plt.figure()

    plt.plot(
        data["Time (day)"],
        data["Theis drawdown (m)"],
        marker="o",
        label="Theis",
    )

    plt.plot(
        data["Time (day)"],
        data["CJ drawdown (m)"],
        marker="s",
        label="Cooper-Jacob",
    )

    plt.xscale("log")
    plt.xlabel("Time (day)")
    plt.ylabel("Drawdown (m)")
    plt.title(f"Theis vs Cooper-Jacob at r = {distance} m")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------

def main():
    """Run the complete analytical pumping-test workflow."""

    print("Reference aquifer properties")
    print("----------------------------")
    print(f"Hydraulic conductivity K = {K:.2f} m/day")
    print(f"Aquifer thickness b = {b:.1f} m")
    print(f"Transmissivity T = {T:.2f} m2/day")
    print(f"Specific storage Ss = {Ss:.2e} 1/m")
    print(f"Storativity S = {S:.2e}")
    print()

    # Generate analytical results
    df = generate_results()

    # Save complete results table
    output_file = "Theis_CJ_results.csv"
    df.to_csv(output_file, index=False)

    print(f"Results saved to: {output_file}")
    print()

    # -------------------------------------------------------------
    # Theis vs Cooper-Jacob error analysis at 50 m
    # -------------------------------------------------------------

    cj_error = calculate_cj_error(
        df=df,
        distance=50,
    )

    mean_error = cj_error["Relative error (%)"].mean()

    print("Theis vs Cooper-Jacob comparison at 50 m")
    print("-----------------------------------------")
    print(
        f"Mean relative error = "
        f"{mean_error:.3f}%"
    )
    print()

    # -------------------------------------------------------------
    # Two-point Cooper-Jacob estimate
    # -------------------------------------------------------------

    T_two_point, t0_two_point, S_two_point = (
        cooper_jacob_two_point_estimate(
            df=df,
            distance=50,
            t1=0.1,
            t2=1.0,
        )
    )

    print("Two-point Cooper-Jacob estimate")
    print("-------------------------------")
    print(
        f"Estimated transmissivity = "
        f"{T_two_point:.3f} m2/day"
    )
    print(
        f"Zero-drawdown intercept t0 = "
        f"{t0_two_point:.6f} day"
    )
    print(
        f"Estimated storativity = "
        f"{S_two_point:.4e}"
    )
    print()

    # -------------------------------------------------------------
    # Regression-based Cooper-Jacob estimate
    # -------------------------------------------------------------

    (
        slope,
        intercept,
        T_fit,
        t0_fit,
        S_fit,
    ) = cooper_jacob_regression(
        df=df,
        distance=50,
    )

    print("Regression-based Cooper-Jacob estimate")
    print("---------------------------------------")
    print(f"Semi-log slope = {slope:.6f}")
    print(f"Regression intercept = {intercept:.6f}")
    print(
        f"Estimated transmissivity = "
        f"{T_fit:.3f} m2/day"
    )
    print(
        f"Zero-drawdown intercept t0 = "
        f"{t0_fit:.6f} day"
    )
    print(
        f"Estimated storativity = "
        f"{S_fit:.4e}"
    )

    # -------------------------------------------------------------
    # Generate plots
    # -------------------------------------------------------------

    plot_theis_at_50m(df)

    plot_theis_all_distances(df)

    plot_theis_vs_cj(
        df=df,
        distance=50,
    )


if __name__ == "__main__":
    main()
