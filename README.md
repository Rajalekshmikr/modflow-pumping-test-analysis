# Pumping-Test Simulation & Aquifer Parameter Recovery using MODFLOW 6

Numerical groundwater-modelling project investigating how aquifer boundaries,
pumping duration and vertical heterogeneity influence pumping-test interpretation
and recovery of transmissivity (T) and storativity (S).

This repository is based on my MSc dissertation in Water, Waste & Environmental
Engineering and combines analytical pumping-test interpretation with numerical
groundwater-flow modelling and Python-based analysis.

---

## Project Objectives

The project investigates:

- agreement between MODFLOW 6 and the analytical Theis solution
- the influence of model-domain size on simulated drawdown
- no-flow, constant-head and general-head boundary effects
- pumping-duration sensitivity in Cooper–Jacob interpretation
- vertical aquifer heterogeneity using controlled three-layer models
- recovery of transmissivity and storativity using Direct Theis and Cooper–Jacob
- computational reproducibility using FloPy

---

## Technical Stack

### Groundwater Modelling
- MODFLOW 6
- ModelMuse
- FloPy

### Pumping-Test Analysis
- Theis solution
- Cooper–Jacob straight-line method
- Transmissivity estimation
- Storativity estimation
- Semi-log drawdown analysis

### Python & Data Analysis
- Python
- NumPy
- pandas
- SciPy
- Matplotlib

---

## Model Overview

The numerical experiments use a synthetic confined aquifer with a centrally
located fully penetrating pumping well.

Key baseline parameters include:

| Parameter | Value |
|---|---:|
| Aquifer thickness | 20 m |
| Hydraulic conductivity | 8.64 m/day |
| Reference transmissivity | 172.8 m²/day |
| Specific storage | 1 × 10⁻⁵ 1/m |
| Reference storativity | 2 × 10⁻⁴ |
| Pumping rate | 50 m³/day |
| Simulation duration | 10 days |
| Observation distances | 50, 100, 200, 500 and 1000 m |

---

## Experimental Workflow

Analytical benchmark  
↓  
Homogeneous MODFLOW baseline  
↓  
Boundary-distance and boundary-type experiments  
↓  
Pumping-duration sensitivity  
↓  
Vertical heterogeneity: H0 → H1 → H2  
↓  
Direct Theis and Cooper–Jacob parameter recovery  
↓  
Error analysis  
↓  
FloPy reproducibility assessment

---

## Key Findings

- Increasing the no-flow model domain from approximately 5 km to 10 km
  reduced Theis–MODFLOW RMSE by approximately **95.9%**.

- The 10 km model produced an overall RMSE of approximately **0.000797 m**.

- Boundary-affected late-time observations increasingly distorted
  Cooper–Jacob parameter estimates.

- Extending the fitting window from 1 to 10 days changed estimated
  transmissivity from approximately **170.95 to 124.75 m²/day**.

- Estimated storativity increased from approximately
  **2.13 × 10⁻⁴ to 8.43 × 10⁻⁴** over the same fitting-window extension.

- Increasing vertical heterogeneity produced stronger layer-specific hydraulic
  responses while bulk transmissivity and storativity recovery remained
  comparatively stable.

- Direct Theis produced approximately **+1% mean T bias and +4% mean S bias**,
  compared with approximately **−2% T bias and +10% mean S bias**
  for Cooper–Jacob.

---

## Repository Development

This repository is being organised into a reproducible technical portfolio
containing:

- analytical pumping-test calculations
- MODFLOW/FloPy model scripts
- synthetic model outputs
- parameter-estimation workflows
- selected figures
- documented modelling methodology

---

## Author

**Rajalekshmi K Raju**

Hydrogeology | Groundwater Modelling | MODFLOW 6 | FloPy | Python | QGIS
