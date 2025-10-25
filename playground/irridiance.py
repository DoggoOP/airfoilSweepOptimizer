from aerosandbox.library import power_solar
import numpy as np
import matplotlib.pyplot as plt


operating_lat = 37.398928
mission_date = 100
N = 180  # Number of discretization points
time = np.linspace(0, 24 * 60 * 60, N)  # s
dt = np.diff(time)[0]  # s
operating_altitude = 1200 # in meters

panel_area = 0.135**2

fluxes = []
for i in range(N):
    solar_flux = power_solar.solar_flux(
        latitude=operating_lat,
        day_of_year=mission_date,
        time=time[i],
        altitude=operating_altitude,
        panel_azimuth_angle=0,
        panel_tilt_angle=0
    ) # W / m^2
    fluxes.append(solar_flux)

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(time / 3600, fluxes, color='tab:blue', label='Solar flux')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Solar Flux (Watts)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

plt.title('Solar flux vs time')
ax1.grid(True)

plt.show()
