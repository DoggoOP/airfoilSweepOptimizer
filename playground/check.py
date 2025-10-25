import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.library import power_solar, propulsion_electric, propulsion_propeller
from aerosandbox.atmosphere import Atmosphere
from lib.aero import calculate_skin_friction


opti=asb.Opti(cache_filename="output/soln1.json")


### CONSTANTS
## Mission
mission_date = 100
lat = 37.398928

temperature_high = 278 # in Kelvin --> this is 60 deg F addition to ISA temperature at 0 meter MSL
operating_altitude = 1200 # in meters
operating_atm = Atmosphere(operating_altitude, temperature_deviation=temperature_high)

## Aerodynamics
# Airfoils
wing_airfoil = asb.Airfoil("sd7037")
tail_airfoil = asb.Airfoil("naca0010")

# Main wing
polyhedral_angle = 10

# Vstab dimensions
vstab_span = 0.3
vstab_chordlen = 0.15

## Power
N = 180  # Number of discretization points
time = np.linspace(0, 24 * 60 * 60, N)  # s
dt = np.diff(time)[0]  # s
battery_voltage = 22.2
encapsulation_eff_hit = 0.1 # Estimated 10% efficieincy loss from encapsulation.
solar_cell_efficiency = 0.243 * (1 - encapsulation_eff_hit)
energy_generation_margin = opti.parameter(value=1.05)
allowable_battery_depth_of_discharge = opti.parameter(value=0.85)  # How much of the battery can you actually use?


### VARIABLES
## Performance
airspeed = opti.variable(init_guess=15, lower_bound=5, upper_bound=30, scale=5, category="airspeed")
thrust_force = opti.variable(init_guess=7, lower_bound=0, scale=7, category="thrust")
power_out_max = opti.variable(init_guess=50, lower_bound=10, scale=50, category="power_out_max")

# Main wing
wingspan = opti.variable(init_guess=6, lower_bound=2, upper_bound=7, scale=6, category="wingspan")
chordlen = opti.variable(init_guess=0.3, scale=1, category="chordlen")
struct_defined_aoa = opti.variable(init_guess=2, lower_bound=0, upper_bound=7, scale=2, category="struct_aoa")
cg_le_dist = opti.variable(init_guess=0.05, lower_bound=0, scale=0.05, category="cg_le_dist")

# Hstab
hstab_span = opti.variable(init_guess=0.5, lower_bound=0.3, upper_bound=1, scale=0.5, category="hstab_span")
hstab_chordlen = opti.variable(init_guess=0.2, lower_bound=0.15, upper_bound=0.4, scale=0.2, category="hstab_chordlen")
hstab_aoa = opti.variable(init_guess=-5, lower_bound=-5, upper_bound=5, scale=5, category="hstab_aoa")

# Body
boom_length = opti.variable(init_guess=2, lower_bound=1.0, upper_bound=4, scale=2, category="boom_length")

## Power
# Propulsion
n_propellers = opti.parameter(1)
propeller_diameter = opti.variable(init_guess=1, lower_bound=0.1, upper_bound=2, scale=1, category="propeller_diameter")
motor_rpm = opti.variable(init_guess=4000, lower_bound=2000, upper_bound=10000, scale=4000, category="motor_rpm")
motor_kv = opti.variable(init_guess=250, lower_bound=150, upper_bound=350, scale=250, category="motor_kv")

# Avionics
n_solar_panels = opti.variable(init_guess=40, lower_bound=10, category="n_solar_panels", scale=40)
battery_capacity = opti.variable(init_guess=450, lower_bound=0, category="battery_capacity", scale=450) # Initial battery energy in Wh. 5Ah*6cells*3.7V/cell=111Wh
battery_states_nondim = opti.variable(n_vars=N, init_guess=0.5, category="battery_states", scale=0.5)


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
vlm = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=asb.OperatingPoint(
        atmosphere=operating_atm, # FIX!
        velocity=airspeed,  # m/s
    ),
)
# abu = asb.AeroBuildup(
#     airplane=airplane,
#     op_point=asb.OperatingPoint(
#         operating_atm,
#         airspeed
#     )
# )
aero = vlm.run_with_stability_derivatives()  # Returns a dictionary
# abu.run()

# VLM does not calcualte parasitic drag, we must add this manually
CD0 = (
    calculate_skin_friction(chordlen, airspeed) * main_wing.area(type="wetted") / main_wing.area()
    + calculate_skin_friction(hstab_chordlen, airspeed) * hor_stabilizer.area(type="wetted") / main_wing.area()
    + calculate_skin_friction(vstab_chordlen, airspeed) * vert_stabilizer.area(type="wetted") / main_wing.area()
    + calculate_skin_friction(0.5, airspeed) * main_fuselage.area_wetted() / main_wing.area()
    + 2 * calculate_skin_friction(0.2, airspeed) * left_pod.area_wetted() / main_wing.area()
)

drag_parasite = 0.5 * 1.29 * airspeed**2 * main_wing.area() * CD0

aero["CD_tot"] = aero["CD"] + CD0
aero["D_tot"] = aero["D"] + drag_parasite
aero["power"] = aero["D_tot"] * airspeed


### PROPULSION
power_out_propulsion_shaft = propulsion_propeller.propeller_shaft_power_from_thrust(
    thrust_force=thrust_force,
    area_propulsive=np.pi / 4 * propeller_diameter ** 2,
    airspeed=airspeed,
    rho=operating_atm.density(),
    propeller_coefficient_of_performance=0.90  # calibrated to QProp output with Dongjoon
)

propeller_tip_mach = 0.36  # From Dongjoon, 4/30/20
propeller_rads_per_sec = propeller_tip_mach * Atmosphere(altitude=1100).speed_of_sound() / (propeller_diameter / 2)
propeller_rpm = propeller_rads_per_sec * 30 / np.pi

motor_rads_per_sec = motor_rpm * 2 * np.pi / 60
motor_torque_per_motor = power_out_propulsion_shaft / motor_rads_per_sec
motor_kv = propeller_rpm / battery_voltage


### POWER
for i in range(N-1):
    solar_flux = power_solar.solar_flux(
        latitude=lat,
        day_of_year=mission_date,
        time=time[i],
        altitude=operating_altitude,
        panel_azimuth_angle=0,
        panel_tilt_angle=0
    ) # W / m^2

    solar_area = n_solar_panels * 0.125**2 # m^2
    power_generated = solar_flux * solar_area * solar_cell_efficiency * energy_generation_margin
    power_used = (power_out_propulsion_shaft + 8)  # 8w to run avionics
    net_energy = (power_generated - power_used) * (dt / 3600)  # Wh
    net_energy_nondim = net_energy / battery_capacity

    battery_update_nondim = np.softmin(battery_states_nondim[i] + net_energy_nondim, battery_capacity, hardness=10)
    opti.subject_to(battery_states_nondim[i+1] == battery_update_nondim)


### WEIGHT
# Power
solar_cell_mass = 0.015 * n_solar_panels
battery_mass = propulsion_electric.mass_battery_pack(battery_capacity)
num_packs = battery_capacity / (5 * 6 * 3.7) # 5 ah, 6 cells, 3.7 V/cell
mass_wires = propulsion_electric.mass_wires(
    wire_length=wingspan / 2,
    max_current=power_out_max / battery_voltage,
    allowable_voltage_drop=battery_voltage * 0.01,
    material="aluminum"
)

# Propulsion
mass_motor_raw = propulsion_electric.mass_motor_electric(
    max_power= power_out_max / n_propellers,
    kv_rpm_volt=motor_kv,
    voltage=battery_voltage
) * n_propellers
mass_esc = propulsion_electric.mass_ESC(power_out_max)
mass_propeller = propulsion_propeller.mass_hpa_propeller(propeller_diameter, power_out_max)

# Structures
foam_volume = main_wing.volume() + hor_stabilizer.volume() + vert_stabilizer.volume()
foam_mass = foam_volume * 30.0  # foam 30kg.m^2
spar_mass = (wingspan / 2 + boom_length) * 0.09  # 90g/m carbon spar 22mm
fuselages_mass = 0.4  # 1kg for all fuselage pods

## Total
weight = 9.81 * (
    solar_cell_mass + 
    battery_mass + 
    foam_mass + 
    spar_mass + 
    fuselages_mass +
    mass_motor_raw + 
    mass_esc + 
    mass_propeller +
    mass_wires
)


### STABILITY
static_margin = (cg_le_dist - aero["x_np"]) / np.softmax(1e-6, main_wing.mean_aerodynamic_chord(), hardness=10)


### CONSTRAINTS
# Mission
opti.subject_to(weight <= 7 * 9.81)

# Performance
opti.subject_to([
    thrust_force > aero["D_tot"],
    power_out_max > power_out_propulsion_shaft,
    power_out_propulsion_shaft > 0
])

# Aerodynamic
opti.subject_to(aero["L"] == weight)
opti.subject_to(wing_airfoil.max_thickness() * chordlen > 0.030)  # must accomodate main spar (22mm)
opti.subject_to(wingspan > 0.13 * n_solar_panels)  # Must be able to fit all of our solar panels 13cm each

# Stability
opti.subject_to(cg_le_dist <= 0.25 * chordlen)
opti.subject_to([
    static_margin > 0.1,
    static_margin < 0.5
])

# Power
opti.subject_to([
    battery_states_nondim > 0,
    battery_states_nondim < allowable_battery_depth_of_discharge,
    battery_states_nondim[0] == 1,
    battery_states_nondim[N-1] == 1
])


### SOLVE
opti.minimize(wingspan)
sol = opti.solve()
opti.save_solution()

print("---Performance---")
print("Airspeed:", sol(airspeed))
print("Total drag:", sol(aero["CD_tot"]))

print("\n--- Wing Dimensions ---")
print("Wingspan:", sol(wingspan))
print("Chordlen:", sol(chordlen))
print("Hstab AoA:", sol(hstab_aoa))
print("Hstab span:", sol(hstab_span))

print("\n--- Power ---")
print("Solar cells #:", sol(n_solar_panels))
print("Solar cell mass:", sol(solar_cell_mass))
print("Battery mass:", sol(battery_mass))
print("Battery pack #:", sol(num_packs))

print("\n--- Propulsion ---")
print("Motor RPM", sol(motor_rpm))
print("Motor KV", sol(motor_kv))
print("Propeller diameter:", sol(propeller_diameter))
print("Motor mass:", sol(mass_motor_raw))
print("ESC mass:", sol(mass_esc))


# for k, v in aero.items():
#     print(f"{k.rjust(4)} : {sol(aero[k])}")

vlm=sol(vlm)
vlm.draw()

airplane=sol(airplane)
airplane.draw()