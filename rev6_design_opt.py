import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.library import power_solar, propulsion_electric, propulsion_propeller
from aerosandbox.atmosphere import Atmosphere


opti=asb.Opti(cache_filename="output/soln1.json")


### CONSTANTS
## Mission
mission_date = 100
operating_lat = 37.398928
togw_max = 7 # kg
temperature_high = 278 # in Kelvin --> this is 60 deg F addition to ISA temperature at 0 meter MSL
operating_altitude = 1200 # in meters
operating_atm = Atmosphere(operating_altitude, temperature_deviation=temperature_high)

## Aerodynamics
# Airfoils
# if doest converge in 100, kill and move on
# loop thourgh airfoil
# min radius 0.085m
# airfoil local camber, local thickness, max thickness
wing_airfoil = asb.Airfoil("sd7037")
tail_airfoil = asb.Airfoil("naca0010")

# Main wing
polyhedral_angle = 10
# 3cm thickness, chord of 40cm approx
# Vstab
vstab_span = 0.3
vstab_chordlen = 0.15

## Structural
structural_mass_markup = 1.2

## Power
battery_voltage = 22.2
N = 180  # Number of discretization points
time = np.linspace(0, 24 * 60 * 60, N)  # s
dt = np.diff(time)[0]  # s
solar_panels_n_rows = 2
solar_encapsulation_eff_hit = 0.1 # Estimated 10% efficieincy loss from encapsulation.
solar_cell_efficiency = 0.243 * (1 - solar_encapsulation_eff_hit)
energy_generation_margin = 1.05
allowable_battery_depth_of_discharge = 0.85  # How much of the battery can you actually use?


### VARIABLES
## Performance
airspeed = opti.variable(init_guess=15, lower_bound=5, upper_bound=30, scale=5, category="airspeed")
togw_design = opti.variable(init_guess=4, lower_bound=1e-3, upper_bound=togw_max, category="togw_max")
power_out_max = opti.variable(init_guess=500, lower_bound=25*16, scale=100, category="power_out_max")

## Propulsion
thrust_cruise = opti.variable(init_guess=4, lower_bound=0, scale=2, category="thrust_cruise")
propeller_n = opti.parameter(2)
propeller_diameter = opti.variable(init_guess=0.5, lower_bound=0.1, upper_bound=2, scale=1, category="propeller_diameter")

## Avionics
solar_panels_n = opti.variable(init_guess=40, lower_bound=10, category="solar_panels_n", scale=40)
battery_capacity = opti.variable(init_guess=450, lower_bound=100, category="battery_capacity", scale=150)  # initial battery energy in Wh
battery_states = opti.variable(n_vars=N, init_guess=500, category="battery_states", scale=100)

## Aerodynamics
# Main wing
wingspan = opti.variable(init_guess=6, lower_bound=2, upper_bound=7, scale=2, category="wingspan")
chordlen = opti.variable(init_guess=0.4, scale=1, category="chordlen")
struct_defined_aoa = opti.variable(init_guess=2, lower_bound=0, upper_bound=7, scale=1, category="struct_aoa")
cg_le_dist = opti.variable(init_guess=0.05, lower_bound=0, scale=0.05, category="cg_le_dist")

# Hstab
hstab_span = opti.variable(init_guess=0.5, lower_bound=0.3, upper_bound=1, scale=0.5, category="hstab_span")
hstab_chordlen = opti.variable(init_guess=0.2, lower_bound=0.15, upper_bound=0.4, scale=0.2, category="hstab_chordlen")
hstab_aoa = opti.variable(init_guess=-5, lower_bound=-5, upper_bound=0, scale=5, category="hstab_aoa")

# Structural
boom_length = opti.variable(init_guess=2, lower_bound=1.0, upper_bound=4, scale=2, category="boom_length")



### GEOMETRIES
main_wing = asb.Wing(
    name="Main Wing",
    symmetric=True,  # Should this wing be mirrored across the XZ plane?
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=[
                0,
                0,
                0,
            ],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=chordlen,
            twist=struct_defined_aoa,  # degrees
            airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
        ),
        asb.WingXSec(  # Mid
            xyz_le=[0.00, 0.5 * wingspan / 2, 0],
            chord=chordlen,
            twist=struct_defined_aoa,
            airfoil=wing_airfoil,
        ),
        asb.WingXSec(  # Tip
            xyz_le=[0.00, wingspan / 2, np.sin(10 * np.pi / 180) * 0.5 * wingspan / 2],
            chord=0.125,
            twist=struct_defined_aoa,
            airfoil=wing_airfoil,
        ),
    ],
)

hor_stabilizer = asb.Wing(
    name="Horizontal Stabilizer",
    symmetric=True,
    xsecs=[
        asb.WingXSec(  # root
            xyz_le=[0, 0, 0],
            chord=hstab_chordlen,
            twist=hstab_aoa,
            airfoil=tail_airfoil,
        ),
        asb.WingXSec(  # tip
            xyz_le=[0.0, hstab_span / 2, 0],
            chord=hstab_chordlen,
            twist=hstab_aoa,
            airfoil=tail_airfoil,
        ),
    ],
).translate([boom_length, 0, 0])

vert_stabilizer = asb.Wing(
    name="Vertical Stabilizer",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=vstab_chordlen,
            twist=0,
            airfoil=tail_airfoil,
        ),
        asb.WingXSec(
            xyz_le=[0.00, 0, vstab_span],
            chord=vstab_chordlen,
            twist=0,
            airfoil=tail_airfoil,
        ),
    ],
).translate([boom_length + hstab_chordlen, 0, 0])

main_fuselage = asb.Fuselage(  # main fuselage
    name="Fuselage",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[0.5 * xi, 0, 0],
            radius=0.6
            * asb.Airfoil("dae51").local_thickness(
                x_over_c=xi
            ),  # half a meter fuselage. Starting at LE and 0.5m forward
        )
        for xi in np.cosspace(0, 1, 30)
    ],
).translate([-0.5, 0, 0])

left_pod = asb.Fuselage(  # left pod fuselage
    name="Fuselage",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[0.2 * xi, 0.75, -0.02],
            radius=0.4
            * asb.Airfoil("dae51").local_thickness(
                x_over_c=xi
            ),  # half a meter fuselage. Starting at LE and 0.5m forward
        )
        for xi in np.cosspace(0, 1, 30)
    ],
)

right_pod = asb.Fuselage(  # right pod fuselage
    name="Fuselage",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[0.2 * xi, -0.75, -0.02],
            radius=0.4
            * asb.Airfoil("dae51").local_thickness(
                x_over_c=xi
            ),  # half a meter fuselage. Starting at LE and 0.5m forward
        )
        for xi in np.cosspace(0, 1, 30)
    ],
)

airplane = asb.Airplane(
    name="rev 6",
    xyz_ref=[0.1 * chordlen, 0, 0],  # CG location
    wings=[main_wing, hor_stabilizer, vert_stabilizer],
    fuselages=[main_fuselage, left_pod, right_pod],
)


### AERODYNAMICS
vlm = asb.AeroBuildup(
    airplane=airplane,
    op_point=asb.OperatingPoint(
        atmosphere=operating_atm, # FIX!
        velocity=airspeed,  # m/s
    ),
)
aero = vlm.run_with_stability_derivatives(
    alpha=True,
    beta=True,
    p=False,
    q=False,
    r=False
)
aero["power"] = aero["D"] * airspeed


### PROPULSION
power_shaft_cruise = propulsion_propeller.propeller_shaft_power_from_thrust(
    thrust_force=thrust_cruise,
    area_propulsive=np.pi / 4 * propeller_diameter ** 2,
    airspeed=airspeed,
    rho=operating_atm.density(),
    propeller_coefficient_of_performance=0.90  # calibrated to QProp output with Dongjoon
)

propeller_tip_mach = 0.36  # From Dongjoon, 4/30/20
propeller_rads_per_sec = propeller_tip_mach * Atmosphere(altitude=1100).speed_of_sound() / (propeller_diameter / 2)
propeller_rpm = propeller_rads_per_sec * 30 / np.pi
motor_kv = propeller_rpm / battery_voltage

thrust_climb = togw_design * 9.81 * np.sind(45) + aero["D"]


### POWER
for i in range(N-1):
    solar_flux = power_solar.solar_flux(
        latitude=operating_lat,
        day_of_year=mission_date,
        time=time[i],
        altitude=operating_altitude,
        panel_azimuth_angle=0,
        panel_tilt_angle=0
    ) # W / m^2

    solar_area = solar_panels_n * 0.125**2 # m^2
    power_generated = solar_flux * solar_area * solar_cell_efficiency / energy_generation_margin
    power_used = (power_shaft_cruise + 8)  # 8w to run avionics
    net_energy = (power_generated - power_used) * (dt / 3600)  # Wh

    battery_update = np.softmin(battery_states[i] + net_energy, battery_capacity, hardness=10)
    opti.subject_to(battery_states[i+1] == battery_update)


### Mass
# Power
mass_solar_cells = 0.015 * solar_panels_n
mass_batteries = propulsion_electric.mass_battery_pack(
    battery_capacity_Wh=battery_capacity,
    battery_pack_cell_fraction=0.95
)
num_packs = battery_capacity / (5 * 6 * 3.7) # 5 ah, 6 cells, 3.7 V/cell
mass_wires = propulsion_electric.mass_wires(
    wire_length=wingspan / 2,
    max_current=power_out_max / battery_voltage,
    allowable_voltage_drop=battery_voltage * 0.01,
    material="aluminum"
)

# Avionics
mass_speedybee = .055
mass_gps = 0.012
mass_telemtry = 0.026
mass_receiver = 0.018
mass_power_board = 0.075 * 2 # 75g estimate for each power board
mass_avionics = mass_speedybee + mass_gps + mass_telemtry + mass_receiver + mass_power_board

# Actuators
mass_servos = .02 * 4 # 20g a servo

# Propulsion
mass_motor_raw = propulsion_electric.mass_motor_electric(
    max_power= power_out_max / propeller_n,
    kv_rpm_volt=motor_kv,
    voltage=battery_voltage
) * propeller_n
mass_motors_mounted = mass_motor_raw * 2
mass_esc = propeller_n * propulsion_electric.mass_ESC(max_power=power_out_max)
mass_propellers = propeller_n * propulsion_propeller.mass_hpa_propeller(
    diameter=propeller_diameter, 
    max_power=power_out_max
)

# Structures
foam_volume = main_wing.volume() + hor_stabilizer.volume() + vert_stabilizer.volume()
mass_foam = foam_volume * 30.0  # foam 30kg.m^2
mass_spar = (wingspan / 2 + boom_length) * 0.09
mass_fuselages = 0.2 # rough estimate


## Total
total_mass = (
    mass_solar_cells + 
    mass_batteries + 
    mass_wires + 
    mass_avionics +
    mass_servos +
    mass_motors_mounted + 
    mass_esc + 
    mass_propellers +
    mass_spar + mass_foam +
    mass_fuselages
)

### STABILITY
static_margin = (cg_le_dist - aero["x_np"]) / np.softmax(1e-6, main_wing.mean_aerodynamic_chord(), hardness=10)


### CONSTRAINTS
# Mission
opti.subject_to(total_mass < togw_design)

# Performance
opti.subject_to(thrust_cruise >= aero["D"])
opti.subject_to(power_out_max >= power_shaft_cruise)
opti.subject_to(power_out_max >= thrust_climb * airspeed)
opti.subject_to(motor_kv >= 150)

# Aerodynamic
opti.subject_to(aero["L"] >= togw_design * 9.81)
opti.subject_to(chordlen >= solar_panels_n_rows * 0.13 + 0.1) # ! Justify this addition of 0.1
opti.subject_to(wing_airfoil.max_thickness() * chordlen >= 0.030)  # must accomodate main spar (22mm)
opti.subject_to(wingspan >= 0.13 * solar_panels_n / solar_panels_n_rows)  # Must be able to fit all of our solar panels 13cm each

# Stability
opti.subject_to(cg_le_dist <= 0.25 * chordlen)
# opti.subject_to(static_margin > 0.1)
# opti.subject_to(static_margin < 0.5)

# Power
opti.subject_to(battery_states > battery_capacity * (1-allowable_battery_depth_of_discharge))
opti.subject_to(battery_states[0] <= battery_states[N-1])


### SOLVE
opti.minimize(wingspan)

try:
    sol = opti.solve()
    opti.save_solution()
except RuntimeError:
    sol = opti.debug

s = lambda x: sol.value(x)

print("---Performance---")
print("Airspeed:", s(airspeed))
print("Thrust cruise:", s(thrust_cruise))
print("Power out max:", s(power_out_max))
print("Mass:", s(total_mass))
print("TOGW Max:", s(togw_max))
print("TOGW Design:", s(togw_design))
print("Weight:", s(togw_design * 9.81))

print("\n--- Aerodynamics ---")
print("CL", s(aero["CL"]))
print("CD", s(aero["CD"]))
print("L/D", s(aero["CL"]/aero["CD"]))
print("Total lift:", s(aero["L"]))
print("Total drag:", s(aero["D"]))

print("\n--- Wing Dimensions ---")
print("Main wing span:", s(wingspan))
print("Main wing chord length:", s(chordlen))
print("Main wing AR:", s(main_wing.aspect_ratio()))
print("Struct defined AoA: ", s(struct_defined_aoa))
print("Hstab AoA:", s(hstab_aoa))
print("Hstab span:", s(hstab_span))
print("Hstab chord:", s(hstab_chordlen))

print("\n--- Structral Dimensions ---")
print("cg_le_dist:", s(cg_le_dist))
print("Boom length:", s(boom_length))
print("Wing mass:", s(mass_foam))
# print("Spar mass:", s(mass_spar))
print("Fuselage mass:", s(mass_fuselages))

print("\n--- Power ---")
print("Solar cells #:", s(solar_panels_n))
print("Solar cell mass:", s(mass_solar_cells))
print("Battery capacity:", s(battery_capacity))
print("Battery mass:", s(mass_batteries))
print("Battery pack #:", s(num_packs))
print("Wire mass:", s(mass_wires))

print("\n--- Power ---")
print("Avionics mass:", mass_avionics)

print("\n--- Propulsion ---")
print("Motor #:", s(propeller_n))
print("Motor RPM:", s(propeller_rpm))
print("Motor KV", s(motor_kv))
print("Propeller diameter:", s(propeller_diameter))
print("Propeller mass:", s(mass_propellers))
print("Motors mass:", s(mass_motor_raw))
print("ESCs mass:", s(mass_esc))


# for k, v in aero.items():
#     print(f"{k.rjust(4)} : {sol(aero[k])}")

# vlm=sol(vlm)
# vlm.draw()

airplane=sol(airplane)
airplane.draw()