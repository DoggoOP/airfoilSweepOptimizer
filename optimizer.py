import os, glob, math, json
from tqdm.auto import tqdm
import numpy as npx
import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.library import power_solar, propulsion_electric, propulsion_propeller
from aerosandbox.atmosphere import Atmosphere

# ---------- CONFIG ----------
AIRFOIL_DIR = "./airfoils"
CSV_OUT = "airfoil_sweep_results.csv"

# Exclude the very leading edge from curvature checks (where curvature is naturally huge)
LE_EXCLUDE_FRACTION = 0.05     # ignore x/c < 0.05 on upper surface
RESAMPLE_N = 400               # curvature sampling points along upper surface
CURVATURE_MAX_1_PER_M = 11.75  # 1/m
RADIUS_MIN_M = 0.085           # m

### CONSTANTS
## Mission
mission_date = 100
operating_lat = 37.398928
togw_max = 7 # kg
temperature_high = 278 # in Kelvin --> this is 60 deg F addition to ISA temperature at 0 meter MSL
operating_altitude = 1200 # in meters
operating_atm = Atmosphere(operating_altitude, temperature_deviation=temperature_high)

# Main wing
polyhedral_angle = 10
# 3cm thickness, chord of 40cm approx
# Vstab
vstab_span = 0.3
vstab_chordlen = 0.15

tail_airfoil = asb.Airfoil("naca0010")

## Structural
structural_mass_markup = 1.2

## Power
battery_voltage = 22.2
N = 180  # Number of discretization points
time = npx.linspace(0.0, 24 * 60 * 60, N)  # s   (real numpy)
dt = float(time[1] - time[0])   # s
solar_panels_n_rows = 2
solar_encapsulation_eff_hit = 0.1 # Estimated 10% efficieincy loss from encapsulation.
solar_cell_efficiency = 0.243 * (1 - solar_encapsulation_eff_hit)
energy_generation_margin = 1.05
allowable_battery_depth_of_discharge = 0.85  # How much of the battery can you actually use?


# For physical curvature/radius, we need a chord. Use your hard lower bound from your constraints:
# chordlen >= solar_panels_n_rows * 0.13 + 0.1
solar_panels_n_rows = 2
CHORD_MAX_M = solar_panels_n_rows * 0.13 + 0.1  # with your current solar_panels_n_rows=2 => 0.36 m
PANEL_EDGE_M = 0.125     # 13 cm square
PANEL_ZONE = (0.18, 0.85)  # panel can live on this x/c range
CHORD_M = CHORD_MAX_M     # your chosen max chord
Lc = PANEL_EDGE_M / CHORD_M  # panel length as fraction of chord

# ---------- AIRFOIL .dat LOADER ----------
def load_airfoil_dat(path):
    """
    Robust .dat reader. Normalizes chord to [0,1] with LE at x=0, TE at x=1.
    Returns: coords (Nx2), asb.Airfoil instance, simple name
    """
    xs, ys = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # skip headers with words
            toks = line.replace(",", " ").split()
            if len(toks) < 2:
                continue
            try:
                x, y = float(toks[0]), float(toks[1])
                xs.append(x); ys.append(y)
            except:
                # header row
                pass

    coords = npx.column_stack([npx.array(xs, dtype=float), npx.array(ys, dtype=float)])
    # Normalize chord: shift so LE x_min -> 0, scale so (x_max - x_min) -> 1
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    chord = x_max - x_min
    if chord <= 0:
        raise ValueError(f"Bad chord in {path}")
    coords_norm = coords.copy()
    coords_norm[:, 0] = (coords[:, 0] - x_min) / chord
    coords_norm[:, 1] = coords[:, 1] / chord  # scale y by chord too

    # Build Airfoil
    af = asb.Airfoil(name=os.path.splitext(os.path.basename(path))[0], coordinates=coords_norm)
    return coords_norm, af, af.name


# ---------- CURVATURE / RADIUS CHECKS ----------
def _circ_radius_three_points(p1, p2, p3):
    # returns absolute radius of circumcircle through three points (in chord units)
    (x1,y1),(x2,y2),(x3,y3) = p1,p2,p3
    A = npx.array([[x1, y1, 1.0],
                   [x2, y2, 1.0],
                   [x3, y3, 1.0]])
    a = npx.linalg.det(A)
    if npx.isclose(a, 0.0):  # nearly collinear => infinite radius
        return npx.inf
    x1s, y1s = x1**2 + y1**2, y1**2 + x1**2  # just to hint structure
    D = npx.linalg.det(npx.array([[x1s, y1, 1.0],
                                  [x2**2 + y2**2, y2, 1.0],
                                  [x3**2 + y3**2, y3, 1.0]]))
    E = npx.linalg.det(npx.array([[x1, x1**2 + y1**2, 1.0],
                                  [x2, x2**2 + y2**2, 1.0],
                                  [x3, x3**2 + y3**2, 1.0]]))
    F = npx.linalg.det(npx.array([[x1, y1, x1**2 + y1**2],
                                  [x2, y2, x2**2 + y2**2],
                                  [x3, y3, x3**2 + y3**2]]))
    # radius (in chord units):
    R_chord = npx.sqrt(D**2 + E**2 - 4*a*F) / (2*npx.abs(a))
    return float(R_chord)

def _upper_surface(coords_norm):
    x = coords_norm[:,0]; y = coords_norm[:,1]
    ile = npx.argmin(x)
    seg1, seg2 = coords_norm[:ile+1], coords_norm[ile:]
    upper = seg1 if npx.nanmean(seg1[:,1]) > npx.nanmean(seg2[:,1]) else seg2
    upper = upper[npx.argsort(upper[:,0])]
    return upper

def min_panel_radius_m(coords_norm):
    upper = _upper_surface(coords_norm)
    x_u, y_u = upper[:,0], upper[:,1]
    # limit to panel placement zone
    mask_zone = (x_u >= PANEL_ZONE[0]) & (x_u <= PANEL_ZONE[1])
    x_u, y_u = x_u[mask_zone], y_u[mask_zone]
    if len(x_u) < 5:
        return 0.0  # force reject

    # resample uniformly in x
    xs = npx.linspace(x_u.min(), x_u.max(), 600)
    ys = npx.interp(xs, x_u, y_u)

    half = 0.5 * Lc
    # centers where a full window is available
    centers = xs[(xs >= xs.min()+half) & (xs <= xs.max()-half)]
    if len(centers) == 0:
        return 0.0

    Rs = []
    for xc in centers[::5]:  # stride to save time
        xL, xR = xc - half, xc + half
        yL = npx.interp(xL, xs, ys)
        yC = npx.interp(xc, xs, ys)
        yR = npx.interp(xR, xs, ys)
        R_c = _circ_radius_three_points((xL,yL),(xc,yC),(xR,yR))  # chord units
        Rs.append(R_c * CHORD_M)  # convert to meters

    if not Rs:
        return 0.0
    # robust statistic: 5th percentile over windows
    return float(npx.nanquantile(npx.array(Rs), 0.05))

def passes_panel_constraints(coords_norm):
    R05 = min_panel_radius_m(coords_norm)
    ok = (R05 >= RADIUS_MIN_M)
    reason = "" if ok else f"R05={R05:.4f} m (need ≥ {RADIUS_MIN_M:.3f} m)"
    return ok, {"panel_R05_m": R05, "reason": reason}


# ---------- WRAPPER AROUND YOUR CURRENT MODEL ----------
def solve_one_airfoil(wing_airfoil_obj, cache_basename="soln"):
    """
    Rebuilds your model for a given wing_airfoil, caps iterations at 200, and returns a dict of results.
    Includes pre-checks to catch problematic airfoils early.
    """
    # ==== EARLY SANITY CHECK: Test if airfoil can produce valid aerodynamics ====
    try:
        # Quick aero test at reasonable conditions
        test_wing = asb.Wing(
            name="Test",
            symmetric=True,
            xsecs=[
                asb.WingXSec(xyz_le=[0, 0, 0], chord=0.4, twist=3, airfoil=wing_airfoil_obj),
                asb.WingXSec(xyz_le=[0, 0.5, 0], chord=0.4, twist=3, airfoil=wing_airfoil_obj),
            ]
        )
        test_atm = Atmosphere(1200, temperature_deviation=60.0 * (5.0/9.0))
        test_vlm = asb.AeroBuildup(
            airplane=asb.Airplane(name="Test", xyz_ref=[0.04, 0, 0], wings=[test_wing]),
            op_point=asb.OperatingPoint(atmosphere=test_atm, velocity=15)
        )
        test_aero = test_vlm.run()
        
        # Check for NaN in critical outputs
        if npx.isnan(test_aero["CL"]) or npx.isnan(test_aero["CD"]) or npx.isnan(test_aero["L"]):
            return dict(
                converged=False,
                fail_reason="Airfoil failed aerodynamic pre-check (NaN in CL/CD/L)",
                **{k: npx.nan for k in ["Airspeed", "Thrust_cruise", "Power_out_max", "Mass", 
                   "TOGW_Design", "Weight", "CL", "CD", "L_over_D", "Lift_total", "Drag_total",
                   "Wing_span", "Wing_chord", "Wing_AR", "Struct_defined_AoA", "Hstab_AoA",
                   "Hstab_span", "Hstab_chord", "cg_le_dist", "Boom_length", "Wing_mass",
                   "Fuselage_mass", "Solar_cells_num", "Solar_cell_mass", "Battery_capacity_Wh",
                   "Battery_mass", "Battery_packs", "Wire_mass", "Motor_num", "Motor_RPM",
                   "Motor_kV", "Propeller_diameter", "Propeller_mass", "Motors_mass", "ESCs_mass"]},
                TOGW_Max=togw_max,
                Avionics_mass=0.261
            )
    except Exception as e:
        return dict(
            converged=False,
            fail_reason=f"Airfoil failed aerodynamic pre-check: {str(e)[:200]}",
            **{k: npx.nan for k in ["Airspeed", "Thrust_cruise", "Power_out_max", "Mass",
               "TOGW_Design", "Weight", "CL", "CD", "L_over_D", "Lift_total", "Drag_total",
               "Wing_span", "Wing_chord", "Wing_AR", "Struct_defined_AoA", "Hstab_AoA",
               "Hstab_span", "Hstab_chord", "cg_le_dist", "Boom_length", "Wing_mass",
               "Fuselage_mass", "Solar_cells_num", "Solar_cell_mass", "Battery_capacity_Wh",
               "Battery_mass", "Battery_packs", "Wire_mass", "Motor_num", "Motor_RPM",
               "Motor_kV", "Propeller_diameter", "Propeller_mass", "Motors_mass", "ESCs_mass"]},
            TOGW_Max=togw_max,
            Avionics_mass=0.261
        )
    
    # ==== BEGIN: your model (unaltered physics/vars; only moved into a function) ====
    opti = asb.Opti(cache_filename=f"output/{cache_basename}.json")

    ### CONSTANTS
    ## Mission
    mission_date_local = mission_date
    operating_lat_local = operating_lat
    togw_max_local = togw_max
    temperature_high_local = temperature_high
    temperature_deviation = 60.0 * (5.0/9.0)
    operating_altitude_local = operating_altitude
    operating_atm_local = Atmosphere(operating_altitude_local, temperature_deviation=temperature_deviation)

    ## Aerodynamics
    wing_airfoil_local = wing_airfoil_obj
    tail_airfoil_local = tail_airfoil

    # Main wing
    polyhedral_angle_local = 10
    # Vstab (from your globals)
    vstab_span_local = vstab_span
    vstab_chordlen_local = vstab_chordlen

    ## Structural
    structural_mass_markup_local = structural_mass_markup

    ## Power
    battery_voltage_local = battery_voltage
    N_local = N
    time_local = time
    dt_local = dt
    solar_panels_n_rows_local = solar_panels_n_rows
    solar_encapsulation_eff_hit_local = solar_encapsulation_eff_hit
    solar_cell_efficiency_local = solar_cell_efficiency
    energy_generation_margin_local = energy_generation_margin
    allowable_battery_depth_of_discharge_local = allowable_battery_depth_of_discharge

    ### VARIABLES (copied from your script)
    airspeed_local = opti.variable(init_guess=15, lower_bound=5, upper_bound=30, scale=5, category="airspeed")
    togw_design_local = opti.variable(init_guess=4, lower_bound=1e-3, upper_bound=togw_max_local, category="togw_max")
    power_out_max_local = opti.variable(init_guess=500, lower_bound=25*16, scale=100, category="power_out_max")

    thrust_cruise_local = opti.variable(init_guess=4, lower_bound=0, scale=2, category="thrust_cruise")
    propeller_n_local = opti.parameter(2)
    propeller_diameter_local = opti.variable(init_guess=0.5, lower_bound=0.1, upper_bound=2, scale=1, category="propeller_diameter")

    solar_panels_n_local = opti.variable(init_guess=40, lower_bound=10, category="solar_panels_n", scale=40)
    battery_capacity_local = opti.variable(init_guess=450, lower_bound=100, category="battery_capacity", scale=150)
    battery_states_local = opti.variable(n_vars=N_local, init_guess=500, lower_bound=0, category="battery_states", scale=100)

    wingspan_local = opti.variable(init_guess=6, lower_bound=2, upper_bound=7, scale=2, category="wingspan")
    chordlen_local = opti.variable(init_guess=0.4, scale=1, category="chordlen")
    struct_defined_aoa_local = opti.variable(init_guess=2, lower_bound=0, upper_bound=7, scale=1, category="struct_aoa")
    cg_le_dist_local = opti.variable(init_guess=0.05, lower_bound=0, scale=0.05, category="cg_le_dist")

    hstab_span_local = opti.variable(init_guess=0.5, lower_bound=0.3, upper_bound=1, scale=0.5, category="hstab_span")
    hstab_chordlen_local = opti.variable(init_guess=0.2, lower_bound=0.15, upper_bound=0.4, scale=0.2, category="hstab_chordlen")
    hstab_aoa_local = opti.variable(init_guess=-5, lower_bound=-5, upper_bound=0, scale=5, category="hstab_aoa")

    boom_length_local = opti.variable(init_guess=2, lower_bound=1.0, upper_bound=4, scale=2, category="boom_length")

    # Geometries (same as your code, swapping in *_local)
    main_wing_local = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=chordlen_local, twist=struct_defined_aoa_local, airfoil=wing_airfoil_local),
            asb.WingXSec(xyz_le=[0.00, 0.5 * wingspan_local / 2, 0], chord=chordlen_local, twist=struct_defined_aoa_local, airfoil=wing_airfoil_local),
            asb.WingXSec(xyz_le=[0.00, wingspan_local / 2, npx.sin(10 * npx.pi / 180) * 0.5 * wingspan_local / 2],
                         chord=0.125, twist=struct_defined_aoa_local, airfoil=wing_airfoil_local),
        ],
    )

    hor_stabilizer_local = asb.Wing(
        name="Horizontal Stabilizer", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=hstab_chordlen_local, twist=hstab_aoa_local, airfoil=tail_airfoil_local),
            asb.WingXSec(xyz_le=[0.0, hstab_span_local / 2, 0], chord=hstab_chordlen_local, twist=hstab_aoa_local, airfoil=tail_airfoil_local),
        ],
    ).translate([boom_length_local, 0, 0])

    vert_stabilizer_local = asb.Wing(
        name="Vertical Stabilizer", symmetric=False,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=vstab_chordlen_local, twist=0, airfoil=tail_airfoil_local),
            asb.WingXSec(xyz_le=[0.00, 0, vstab_span_local], chord=vstab_chordlen_local, twist=0, airfoil=tail_airfoil_local),
        ],
    ).translate([boom_length_local + hstab_chordlen_local, 0, 0])

    main_fuselage_local = asb.Fuselage(
        name="Fuselage",
        xsecs=[
            asb.FuselageXSec(xyz_c=[0.5 * xi, 0, 0], radius=0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi))
            for xi in np.cosspace(0, 1, 30)
        ],
    ).translate([-0.5, 0, 0])

    left_pod_local = asb.Fuselage(
        name="Fuselage",
        xsecs=[
            asb.FuselageXSec(xyz_c=[0.2 * xi, 0.75, -0.02], radius=0.4 * asb.Airfoil("dae51").local_thickness(x_over_c=xi))
            for xi in np.cosspace(0, 1, 30)
        ],
    )

    right_pod_local = asb.Fuselage(
        name="Fuselage",
        xsecs=[
            asb.FuselageXSec(xyz_c=[0.2 * xi, -0.75, -0.02], radius=0.4 * asb.Airfoil("dae51").local_thickness(x_over_c=xi))
            for xi in np.cosspace(0, 1, 30)
        ],
    )

    airplane_local = asb.Airplane(
        name="rev 6",
        xyz_ref=[0.1 * chordlen_local, 0, 0],
        wings=[main_wing_local, hor_stabilizer_local, vert_stabilizer_local],
        fuselages=[main_fuselage_local, left_pod_local, right_pod_local],
    )

    vlm_local = asb.AeroBuildup(
        airplane=airplane_local,
        op_point=asb.OperatingPoint(atmosphere=operating_atm_local, velocity=airspeed_local),
    )
    aero_local = vlm_local.run_with_stability_derivatives(alpha=True, beta=True, p=False, q=False, r=False)
    aero_local["power"] = aero_local["D"] * airspeed_local

    power_shaft_cruise_local = propulsion_propeller.propeller_shaft_power_from_thrust(
        thrust_force=thrust_cruise_local,
        area_propulsive=npx.pi / 4 * propeller_diameter_local ** 2,
        airspeed=airspeed_local,
        rho=operating_atm_local.density(),
        propeller_coefficient_of_performance=0.90
    )

    propeller_tip_mach_local = 0.36
    propeller_rads_per_sec_local = propeller_tip_mach_local * Atmosphere(altitude=1100).speed_of_sound() / (propeller_diameter_local / 2)
    propeller_rpm_local = propeller_rads_per_sec_local * 30 / npx.pi
    motor_kv_local = propeller_rpm_local / battery_voltage_local

    thrust_climb_local = togw_design_local * 9.81 * np.sind(45) + aero_local["D"]

    # Battery dynamics loop (as-is)
    for i in range(N_local - 1):
        solar_flux = power_solar.solar_flux(
            latitude=operating_lat_local, day_of_year=mission_date_local, time=time_local[i],
            altitude=operating_altitude_local, panel_azimuth_angle=0, panel_tilt_angle=0
        )
        solar_area = solar_panels_n_local * 0.125 ** 2
        power_generated = solar_flux * solar_area * solar_cell_efficiency_local / energy_generation_margin_local
        power_used = (power_shaft_cruise_local + 8)
        net_energy = (power_generated - power_used) * (dt_local / 3600)
        battery_update = np.softmin(battery_states_local[i] + net_energy,
                             battery_capacity_local, hardness=10)
        opti.subject_to(battery_states_local[i+1] == battery_update)  # FIX: Connect battery states!
    opti.subject_to(battery_states_local[0] == battery_capacity_local)
    opti.subject_to(battery_states_local <= battery_capacity_local)

    # Mass model (as-is)
    mass_solar_cells = 0.015 * solar_panels_n_local
    mass_batteries = propulsion_electric.mass_battery_pack(battery_capacity_Wh=battery_capacity_local, battery_pack_cell_fraction=0.95)
    num_packs = battery_capacity_local / (5 * 6 * 3.7)
    mass_wires = propulsion_electric.mass_wires(
        wire_length=wingspan_local / 2,
        max_current=power_out_max_local / battery_voltage_local,
        allowable_voltage_drop=battery_voltage_local * 0.01,
        material="aluminum"
    )
    mass_speedybee = .055
    mass_gps = 0.012
    mass_telemtry = 0.026
    mass_receiver = 0.018
    mass_power_board = 0.075 * 2
    mass_avionics = mass_speedybee + mass_gps + mass_telemtry + mass_receiver + mass_power_board
    mass_servos = .02 * 4
    mass_motor_raw = propulsion_electric.mass_motor_electric(
        max_power= power_out_max_local / propeller_n_local,
        kv_rpm_volt=motor_kv_local,
        voltage=battery_voltage_local
    ) * propeller_n_local
    mass_motors_mounted = mass_motor_raw * 2
    mass_esc = propeller_n_local * propulsion_electric.mass_ESC(max_power=power_out_max_local)
    mass_propellers = propeller_n_local * propulsion_propeller.mass_hpa_propeller(
        diameter=propeller_diameter_local, 
        max_power=power_out_max_local
    )
    foam_volume = main_wing_local.volume() + hor_stabilizer_local.volume() + vert_stabilizer_local.volume()
    mass_foam = foam_volume * 30.0
    mass_spar = (wingspan_local / 2 + boom_length_local) * 0.09
    mass_fuselages = 0.2

    total_mass = (
        mass_solar_cells + mass_batteries + mass_wires + mass_avionics + mass_servos +
        mass_motors_mounted + mass_esc + mass_propellers + mass_spar + mass_foam + mass_fuselages
    )

    static_margin = (cg_le_dist_local - aero_local["x_np"]) / np.softmax(1e-6, main_wing_local.mean_aerodynamic_chord(), hardness=10)

    # Constraints (as-is)
    opti.subject_to(total_mass < togw_design_local)
    opti.subject_to(thrust_cruise_local >= aero_local["D"])
    opti.subject_to(power_out_max_local >= power_shaft_cruise_local)
    opti.subject_to(power_out_max_local >= thrust_climb_local * airspeed_local)
    opti.subject_to(motor_kv_local >= 150)
    opti.subject_to(aero_local["L"] >= togw_design_local * 9.81)
    opti.subject_to(chordlen_local >= solar_panels_n_rows_local * 0.13 + 0.1)
    opti.subject_to(wing_airfoil_local.max_thickness() * chordlen_local >= 0.030)
    opti.subject_to(wingspan_local >= 0.13 * solar_panels_n_local / solar_panels_n_rows_local)
    # NOTE: No upper bound on chord in original code
    opti.subject_to(cg_le_dist_local <= 0.25 * chordlen_local)
    opti.subject_to(battery_states_local > battery_capacity_local * (1-allowable_battery_depth_of_discharge_local))
    opti.subject_to(battery_states_local[0] <= battery_states_local[N_local-1])

    # Objective (your original: minimize span)
    opti.minimize(wingspan_local)

    # ---- Solver options: handle NaN and convergence issues ----
    try:
        opti.solver_options["ipopt"] = opti.solver_options.get("ipopt", {})
        ip = opti.solver_options["ipopt"]
        ip["max_iter"] = 200                     # Reduced from 250 - fail faster on bad airfoils
        ip["max_cpu_time"] = 120.0              # 2 minute timeout per airfoil
        ip["tol"] = 1e-6
        ip["acceptable_tol"] = 1e-3              # Relaxed tolerance
        ip["acceptable_iter"] = 15               # More patience for acceptable solutions
        ip["mu_strategy"] = "adaptive"
        ip["hessian_approximation"] = "limited-memory"
        ip["print_level"] = 0                    # Suppress output
        ip["sb"] = "yes"                         # Suppress banner
        ip["bound_relax_factor"] = 1e-8         
        ip["honor_original_bounds"] = "yes"     
        ip["nlp_scaling_method"] = "gradient-based"
        ip["diverging_iterates_tol"] = 1e10     # Detect divergence early
        ip["skip_finalize_solution_call"] = "yes"  # Skip final checks on failure

    except Exception:
        pass

    try:
        sol = opti.solve()
        s = lambda x: sol.value(x)
        converged = True
        fail_reason = ""
    except Exception as e:
        converged = False
        fail_reason = str(e)[:500]
        s = lambda x: npx.nan  # keep types simple for CSV

    out = dict(
        converged=converged,
        fail_reason=fail_reason,
        Airspeed=s(airspeed_local),
        Thrust_cruise=s(thrust_cruise_local),
        Power_out_max=s(power_out_max_local),
        Mass=s(total_mass),
        TOGW_Max=togw_max_local,
        TOGW_Design=s(togw_design_local),
        Weight=s(togw_design_local * 9.81),
        CL=s(aero_local["CL"]),
        CD=s(aero_local["CD"]),
        L_over_D=s(aero_local["CL"]/aero_local["CD"]) if converged else npx.nan,
        Lift_total=s(aero_local["L"]),
        Drag_total=s(aero_local["D"]),
        Wing_span=s(wingspan_local),
        Wing_chord=s(chordlen_local),
        Wing_AR=s(main_wing_local.aspect_ratio()),
        Struct_defined_AoA=s(struct_defined_aoa_local),
        Hstab_AoA=s(hstab_aoa_local),
        Hstab_span=s(hstab_span_local),
        Hstab_chord=s(hstab_chordlen_local),
        cg_le_dist=s(cg_le_dist_local),
        Boom_length=s(boom_length_local),
        Wing_mass=s(mass_foam),
        Fuselage_mass=s(mass_fuselages),
        Solar_cells_num=s(solar_panels_n_local),
        Solar_cell_mass=s(mass_solar_cells),
        Battery_capacity_Wh=s(battery_capacity_local),
        Battery_mass=s(mass_batteries),
        Battery_packs=s(num_packs),
        Wire_mass=s(mass_wires),
        Avionics_mass=mass_avionics,
        Motor_num=s(propeller_n_local),
        Motor_RPM=s(propeller_rpm_local),
        Motor_kV=s(motor_kv_local),
        Propeller_diameter=s(propeller_diameter_local),
        Propeller_mass=s(mass_propellers),
        Motors_mass=s(mass_motor_raw),
        ESCs_mass=s(mass_esc),
    )
    return out
    # ==== END your model ====


# ---------- MAIN SWEEP ----------
def main():
    import warnings
    warnings.filterwarnings('ignore')  # Suppress Python warnings
    
    # Suppress CasADi warnings by redirecting stderr temporarily during solves
    import sys
    from contextlib import contextmanager
    
    @contextmanager
    def suppress_stderr():
        """Context manager to suppress stderr output"""
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            yield
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
    
    rows = []
    rejections = []
    os.makedirs("output", exist_ok=True)

    airfoil_paths = sorted(glob.glob(os.path.join(AIRFOIL_DIR, "*.dat")))
    accepted = 0
    rejected = 0
    converged_cnt = 0
    failed_aero_cnt = 0

    print(f"Starting airfoil sweep: {len(airfoil_paths)} airfoils to process")
    print("=" * 80)

    for dat_path in tqdm(airfoil_paths, desc="Airfoils", unit="foil"):
        try:
            coords_norm, af, name = load_airfoil_dat(dat_path)
        except Exception as e:
            rejections.append({"airfoil": os.path.basename(dat_path), "reason": f"load_failed: {e}"})
            rejected += 1
            continue

        ok, stats = passes_panel_constraints(coords_norm)
        if not ok:
            rejections.append({"airfoil": name, **stats})
            rejected += 1
            continue

        # Run optimizer for this airfoil (suppress CasADi warnings)
        with suppress_stderr():
            result = solve_one_airfoil(af, cache_basename=f"{name}_soln")
        
        result["airfoil"] = name
        
        # Track different failure types
        if result["converged"]:
            converged_cnt += 1
        elif "aerodynamic pre-check" in result.get("fail_reason", ""):
            failed_aero_cnt += 1
        
        # attach precheck stats
        result["min_radius_m"] = stats.get("panel_R05_m", npx.nan) if "panel_R05_m" in stats else stats.get("min_radius_m", npx.nan)
        result["max_curv_1_per_m"] = stats.get("max_curv_1_per_m", npx.nan)
        rows.append(result)
        accepted += 1
        tqdm.write(f"{name}: {'OK' if result['converged'] else 'FAIL'} {result.get('fail_reason','')[:80]}")



    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)

    # Optional: write a JSON of rejections for inspection
    with open("airfoil_rejections.json", "w") as f:
        json.dump(rejections, f, indent=2)

    print()
    print("=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"Total airfoils processed:      {len(airfoil_paths)}")
    print(f"Rejected by curvature check:   {len(rejections)}")
    print(f"Passed to optimizer:           {len(rows)}")
    print(f"  - Converged successfully:    {converged_cnt}")
    print(f"  - Failed aero pre-check:     {failed_aero_cnt}")
    print(f"  - Failed optimization:       {len(rows) - converged_cnt - failed_aero_cnt}")
    print()
    print(f"Success rate: {100*converged_cnt/len(rows):.1f}% of airfoils passed to optimizer")
    print()
    print(f"Results saved to: {CSV_OUT}")
    print(f"Rejections saved to: airfoil_rejections.json")

if __name__ == "__main__":
    # Optional: pick a start method explicitly (macOS sometimes benefits)
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn")   # or "forkserver"; avoid raw "fork" with heavy C libs
    except Exception:
        pass
    main()