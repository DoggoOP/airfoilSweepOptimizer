import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.library.aerodynamics as lib_aero
from aerosandbox.library import mass_structural as lib_mass_struct
from aerosandbox.library import power_solar as lib_solar
from aerosandbox.library import propulsion_electric as lib_prop_elec
from aerosandbox.library import propulsion_propeller as lib_prop_prop
import aerosandbox.tools.units as u
import copy
from pathlib import Path


opti = asb.Opti(
    freeze_style='float',
    # variable_categories_to_freeze='all'
)

### CONSTANTS
## Mass & mass related
# Power 
liion_21700_cell_mass = 70 # grams
solar_panel_with_busbar_solder_mass = 15 # grams
mppt_mass = 200 # grams

# Avionics
speedybee_f405_mass = 55 # grams
gps_bn880_mass = 12 # grams
telemtry_with_wire_mass = 26 # grams
radio_receiver_with_wire_mass = 18 # grams
wiring_mass = 50 # grams
servos_mass = 50 # grams

# Propulsion
esc_mass = 25 # ! grams, may change ESC
motor_410kv_mass = 277 # grams
propeller_mass = 52 # grams

# Structures
fuselage_mass = 100 # grams
motor_pods_mass = 60 # grams
carbon_22_per_meter_mass = 90 # grams / m
carbon_20_per_meter_mass = 80 # grams / m

# Wing
ngx_foam_density = 30.02 # kg / m^3




# PARAMETERS
n_panels_spanwise_inboard = opti.parameter(12)  # number of panels per side
n_panels_chordwise_inboard = opti.parameter(1)
n_panels_spanwise_outboard = opti.parameter(12)  # number of panels per side
n_panels_chordwise_outboard = opti.parameter(1)




# MASS
design_mass_TOGW = opti.variable(
    init_guess=5,
    lower_bound=1e-3,
)


# get # panels for certain power

# get # batteries for certain power



## Mass
# 