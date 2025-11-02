# Solar Aircraft Design Optimization

An automated design optimization tool for solar-powered aircraft using AeroSandbox. This project optimizes aircraft geometry, propulsion, and power systems to minimize wingspan while meeting energy and performance requirements.

## Overview

This toolkit consists of three main components:
1. **Single Aircraft Optimization** - Optimize a single design with a chosen airfoil
2. **Airfoil Sweep** - Automatically test multiple airfoils and find the best performer
3. **Results Visualization** - Plot battery states and analyze performance

## Requirements

### Python Environment
- Python 3.10 or 3.11 (Python 3.13+ not supported due to dependency issues)
- Conda or venv for environment management

### Dependencies
```bash
# Core packages
aerosandbox
numpy
scipy<=1.11  # Important: newer versions incompatible
pandas
matplotlib
tqdm
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create environment with Python 3.10
conda create -n aircraft_opt python=3.10

# Activate environment
conda activate aircraft_opt

# Install dependencies
conda install -c conda-forge numpy scipy matplotlib pandas tqdm
pip install aerosandbox
```

### Option 2: Using venv

```bash
# Create virtual environment
python3.10 -m venv aircraft_env

# Activate (Linux/Mac)
source aircraft_env/bin/activate

# Activate (Windows)
aircraft_env\Scripts\activate

# Install dependencies
pip install aerosandbox numpy scipy==1.11 pandas matplotlib tqdm
```

## Project Structure

```
.
├── rev6_design_opt.py      # Single aircraft optimization
├── optimizer.py            # Airfoil sweep optimizer
├── plot.py                 # Battery state visualization
├── airfoils/               # Directory containing .dat airfoil files
│   ├── sd7037.dat
│   ├── naca0012.dat
│   └── ...
└── output/                 # Generated optimization results
    ├── soln1.json
    ├── airfoil_sweep_results.csv
    └── ...
```

## Usage

### 1. Single Aircraft Optimization

Optimize a single aircraft design with a specific airfoil:

```bash
python rev6_design_opt.py
```

**What it does:**
- Uses the `sd7037` airfoil (configurable in script)
- Optimizes for minimum wingspan
- Ensures 24-hour energy balance
- Saves results to `output/soln1.json`
- Displays detailed performance metrics

**Key outputs:**
- Performance: airspeed, thrust, power, mass
- Aerodynamics: CL, CD, L/D ratio
- Geometry: wingspan, chord, aspect ratio
- Power system: battery capacity, solar panels
- Propulsion: motor specs, propeller size

### 2. Airfoil Sweep Optimization

Test multiple airfoils to find the best design:

```bash
python optimizer.py
```

**What it does:**
- Loads all `.dat` files from `./airfoils/` directory
- Pre-filters airfoils based on curvature constraints
- Runs optimization for each valid airfoil
- Saves results to `airfoil_sweep_results.csv`
- Generates `airfoil_rejections.json` with failed airfoils

**Airfoil constraints checked:**
- Minimum radius of curvature: 0.085 m
- Panel placement zone: 18-85% chord
- Adequate space for 0.125m solar panels

### 3. Visualize Results

Plot battery energy over a 24-hour mission:

```bash
python plot.py
```

Opens a matplotlib window showing battery state of charge throughout the day.

## Configuration

### Mission Parameters (in scripts)

```python
# Mission
mission_date = 100              # Day of year (1-365)
operating_lat = 37.398928       # Latitude (degrees)
operating_altitude = 1200       # Altitude (meters)
temperature_high = 278          # Temperature deviation (K)

# Constraints
togw_max = 7                    # Max takeoff weight (kg)
solar_panels_n_rows = 2         # Solar panel rows
allowable_battery_depth_of_discharge = 0.85
```

### Airfoil Selection

**For single optimization (`rev6_design_opt.py`):**
```python
wing_airfoil = asb.Airfoil("sd7037")  # Change airfoil here
```

**For sweep (`optimizer.py`):**
Place `.dat` files in `./airfoils/` directory. The script will automatically:
1. Load all airfoils
2. Check geometric constraints
3. Optimize valid candidates

### Solver Settings

The optimizer uses IPOPT with the following key settings:
- Max iterations: 250
- Tolerance: 1e-6
- Acceptable tolerance: 1e-3
- Strategy: Adaptive barrier parameter

## Understanding the Outputs

### Console Output (Single Optimization)

```
---Performance---
Airspeed: 12.5 m/s
Thrust cruise: 3.2 N
Power out max: 450 W
Mass: 6.1 kg
TOGW Design: 6.5 kg

--- Aerodynamics ---
CL: 1.15
CD: 0.045
L/D: 25.6
```

### CSV Output (Airfoil Sweep)

Columns include:
- `airfoil`: Airfoil name
- `converged`: True/False optimization success
- `Wing_span`: Optimized wingspan (m)
- `L_over_D`: Lift-to-drag ratio
- `Mass`: Total aircraft mass (kg)
- `Battery_capacity_Wh`: Battery size
- `Solar_cells_num`: Number of solar panels
- And many more...

### Rejection Reasons

Common rejection reasons in `airfoil_rejections.json`:
- `R05 < 0.085m`: Curvature too high for solar panel mounting
- `load_failed`: Corrupted or invalid .dat file
- `converged=False`: Optimization failed to find solution

## Design Variables

The optimizer adjusts the following variables:

**Performance:**
- Airspeed (5-30 m/s)
- Cruise thrust (N)
- Max power output (W)

**Geometry:**
- Wingspan (2-7 m)
- Wing chord length (m)
- Structural angle of attack (0-7°)
- Horizontal stabilizer dimensions
- Tail boom length (1-4 m)

**Power System:**
- Number of solar panels
- Battery capacity (Wh)
- Propeller diameter (0.1-2 m)

## Troubleshooting

### Import Error: cannot import 'cumtrapz'

**Problem:** SciPy version too new (>1.11)

**Solution:**
```bash
conda install scipy=1.10 -c conda-forge
```

### Optimization doesn't converge

**Possible causes:**
1. Airfoil has extreme geometry
2. Constraints are too restrictive
3. Initial guesses are far from solution

**Solutions:**
- Try a different airfoil
- Relax constraints (e.g., increase `togw_max`)
- Adjust initial guesses in the script

### FileNotFoundError: airfoils/

**Solution:**
```bash
mkdir airfoils
# Add .dat files to this directory
```

### All airfoils rejected

Check `airfoil_rejections.json` for reasons. Common issues:
- High camber airfoils may fail curvature checks
- Very thick airfoils may cause optimization issues

## Airfoil File Format

Airfoil `.dat` files should be in standard Selig or Lednicer format:

```
NACA 0012
1.00000  0.00000
0.95000  0.00555
...
0.00000  0.00000
...
```

The script automatically normalizes coordinates to [0,1] chord.

## Advanced Usage

### Modify Optimization Objective

Change the objective function in the scripts:

```python
# Minimize wingspan (current)
opti.minimize(wingspan)

# Alternative: Minimize mass
opti.minimize(total_mass)

# Alternative: Maximize L/D
opti.minimize(-aero["CL"]/aero["CD"])
```

### Add Custom Constraints

```python
# Example: Limit aspect ratio
opti.subject_to(main_wing.aspect_ratio() <= 20)

# Example: Minimum L/D ratio
opti.subject_to(aero["CL"]/aero["CD"] >= 20)
```

### Parallel Processing

For large airfoil sweeps, consider modifying `optimizer.py` to use multiprocessing (commented in the script).

## Output Files

- `output/soln1.json` - Detailed optimization solution
- `airfoil_sweep_results.csv` - Comparison of all tested airfoils
- `airfoil_rejections.json` - Pre-filtered airfoils with reasons
- `output/<airfoil>_soln.json` - Individual solutions for each airfoil

## License

[Add your license here]

## Contributors

[Add contributors here]

## References

- AeroSandbox: https://github.com/peterdsharpe/AeroSandbox
- UIUC Airfoil Database: https://m-selig.ae.illinois.edu/ads/coord_database.html

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review AeroSandbox documentation
3. Verify all dependencies are correct versions