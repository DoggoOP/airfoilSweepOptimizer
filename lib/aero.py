from aerosandbox import numpy as np

def calculate_skin_friction(length, asp):
    reynolds = 1.29 * asp * length / 1.78e-5
    cf = 0.455 / (np.log10(reynolds) ** 2.58)  # Prandtl-Schlichting’s approximation
    return cf