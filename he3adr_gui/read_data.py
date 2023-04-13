import matplotlib.pyplot as plt
import numpy as np
import re

with open('/data/kpac/temp_data/2015-03-31') as f:
    content = f.readlines()

n = len(content)
r = re.compile('[\t]+')
time = np.zeros(n)
adr = np.zeros(n)
pot = np.zeros(n)
for i in range(n):
    fields = r.split(content[i])
    time[i] = float(fields[0])
    adr[i] = float(fields[6])
    pot[i] = float(fields[4])
    
plt.plot(time, adr, label='ADR')
plt.plot(time, pot, label='Pot')

plt.xlabel('time (s)')
plt.ylabel('Temperature (K)')

plt.title("Simple Plot")

plt.legend()

plt.show()
