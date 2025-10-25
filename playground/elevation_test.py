import matplotlib.pyplot as plt
from aerosandbox.library import power_solar
import aerosandbox.numpy as np

N = 100  # Number of discretization points
time = np.linspace(0, 24*60*60, N) # s
dt = np.diff(time)[0] # s
mission_date = 100
lat = 37.7749

elevation_arr = []
power_arr = []
battery_state = [800]  # Start with a fully charged 10 Wh battery
battery_cap = 1000

for i in range(N):
    elevation = power_solar.solar_elevation_angle(lat, mission_date, time[i])
    power = np.softmax(0, 6 * np.sin(np.deg2rad(elevation)), hardness=10) * 0.55 * 50  # W per panel
    net_energy = (power - 60) * (dt / 3600) # Convert W to Wh (since time is in seconds)
    new_battery_state = np.softmin(battery_state[-1] + net_energy, battery_cap, hardness=10)
    
    power_arr.append(power)
    elevation_arr.append(elevation)
    battery_state.append(new_battery_state)

time_hours = time / 3600

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(time_hours, elevation_arr, color='tab:blue', label='Solar Elevation Angle')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Solar Elevation Angle (degrees)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(time_hours, power_arr, color='tab:orange', label='Power Generated')
ax2.set_ylabel('Panel Amps', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset the battery axis for clarity
ax3.plot(time_hours, battery_state[1:], color='tab:green', label='Battery State (Wh)')
ax3.set_ylabel('Battery State (Wh)', color='tab:green')
ax3.tick_params(axis='y', labelcolor='tab:green')

plt.title('Solar Elevation Angle, Power Wattage, and Battery State vs Time')
ax1.grid(True)

plt.show()
