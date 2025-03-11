# %% imports
import numpy as np
import matplotlib.pyplot as plt

# %%
YOLK_RADIUS_CM = 2
EGG_LENGTH_CM = 8
B = 0.09  # Egg shape parameter
N_X = 100
N_Y = 100


def egg_curve_squared(a: float, b: float, x: float | np.ndarray) -> float | np.ndarray:
    return x * 0.5 * ((a - b) - 2 * x + np.sqrt(4 * b * x + (a - b) ** 2))


egg_length_metres = EGG_LENGTH_CM / 100
yolk_radius_metres = YOLK_RADIUS_CM / 100

xx = np.arange(start=0, stop=egg_length_metres, step=egg_length_metres / N_X)
yy = np.arange(start=0, stop=egg_length_metres, step=egg_length_metres / N_Y)

grid = np.meshgrid(xx, yy)


egg_domain = np.zeros(shape=(N_X, N_Y))
for i, _ in enumerate(egg_domain):
    for j, _ in enumerate(egg_domain[i]):
        x = xx[i]
        y = yy[j]

        if y**2 <= egg_curve_squared(a=egg_length_metres, b=B, x=x):
            egg_domain[i, j] = 1

# %% plot egg domain
plt.figure()
plt.imshow(egg_domain)
plt.show()
