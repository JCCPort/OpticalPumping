import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('QT5Agg')


def ra(x, t):
    return (x ** 2) / ((t / 2) ** 2 + x ** 2)


vals = np.linspace(0, 10, 100)

for m in range(0, 5):
    plt.plot(vals, ra(vals, m), label=r'$\gamma = {}$'.format(m), lw=2)

plt.legend()
plt.ylabel(r'$\frac{\Omega^2_{ab}}{(\gamma/2)^2+\Omega^2_{ab}}$', fontsize=20)
plt.xlabel(r'$\Omega_{ab}$', fontsize=16)
plt.show()
