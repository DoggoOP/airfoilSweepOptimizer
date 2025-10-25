import json
import numpy as np
import matplotlib.pyplot as plt


with open('output/soln1.json') as user_file:
  file_contents = user_file.read()
parsed_json = json.loads(file_contents)
battery_states = parsed_json["power"][2]

# Create a time array for 24 hours with 100 steps
time = np.linspace(0, 24, len(battery_states))  # time in hours

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(time, battery_states, label='Battery Energy (Wh)', color='teal')
plt.xlabel("Time (hours)")
plt.ylabel("Battery Energy (Wh)")
plt.title("Battery Energy vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
