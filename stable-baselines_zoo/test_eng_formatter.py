# import matplotlib.pyplot as plt
# import numpy as np

# from matplotlib.ticker import EngFormatter

# ax = plt.subplot(111)
# ax.set_xscale('log')

# formatter = EngFormatter(places=1)
# ax.xaxis.set_major_formatter(formatter)

# xs = np.logspace(1, 9, 100)
# ys = (0.8 + 0.4 * np.random.uniform(size=100)) * np.log10(xs)**2
# ax.plot(xs, ys)

# plt.show()


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1000, 1001, 100)
y = np.linspace(1e-9, 1e9, 100)

fig, ax = plt.subplots()
ax.plot(x, y)

# ax.ticklabel_format(useOffset=True)
ax.ticklabel_format(style='scientific')

plt.show()
