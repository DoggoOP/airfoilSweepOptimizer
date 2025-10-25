from aerosandbox.library import propulsion_electric

check = propulsion_electric.mass_motor_electric(
    max_power= 500,
    kv_rpm_volt=200,
    voltage=25
)

print(check)