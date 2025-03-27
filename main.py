# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import boil_an_egg.utils as bae

# %%
# Simulation parameters
EGG_LENGTH_METRES = 7 / 100
YOLK_RADIUS_METRES = 1.5 / 100
WATER_TEMPERATURE_CELSIUS = 100
B = 0.05  # Egg shape parameter
nx, ny = 10, 10  # Number of grid points

# Lx, Ly domain dimensions
Lx = EGG_LENGTH_METRES  # Domain dimensions = egg length
# y dimension depends on how wide the egg is
Ly = float(np.max(np.sqrt(bae.egg_curve_squared(a=Lx, b=B, x=np.linspace(0, Lx, nx)))))

dx = Lx / (nx - 1)  # Spatial step size in x
dy = Ly / (ny - 1)  # Spatial step size in y
tmax = 60 * 5  # Maximum simulation time
dt = 1  # Time step size

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Separate white and yolk
egg_domain = bae.create_egg_domain(
    nx=nx,
    ny=ny,
    Lx=Lx,
    Ly=Ly,
    yolk_radius_metres=YOLK_RADIUS_METRES,
    B_EGG_SHAPE_PARAM=B,
)

# Egg is not square
# => some gridpoints lie outside the egg.
# => Fewer equations needed
# => Need a way to map equation number (position in system of eqs.) to point in egg.
egg_to_equation_system_map = bae.compute_egg_to_equation_system_map(
    nx=nx, ny=ny, egg_domain=egg_domain
)
map_from_mesh_cell_numbers_to_coords = bae.map_mesh_cell_numbers_to_coords(
    egg_to_equation_system_map
)

map_from_coords_to_mesh_cell_numbers = bae.invert_dictionary(
    dictionary=map_from_mesh_cell_numbers_to_coords, are_values_unique=True
)

unstructured_egg_domain = bae.create_unstructured_array_from_structured_array(
    structured_array=egg_domain,
    map_from_mesh_cell_numbers_to_coords=map_from_mesh_cell_numbers_to_coords,
)

nearest_neighbors = bae.get_nearest_neighbors(
    nx=nx,
    ny=ny,
    map_from_mesh_cell_numbers_to_coords=map_from_mesh_cell_numbers_to_coords,
    egg_to_equation_system_map=egg_to_equation_system_map,
)
egg_boundary_mesh_cells = bae.get_egg_boundary_mesh_cells(
    nearest_neighbors=nearest_neighbors
)

# Initial condition
u_init = (273 + 20) * np.ones(len(nearest_neighbors))

# Plot egg domain
plt.figure()
plt.imshow(egg_domain, extent=[0, Ly, 0, Lx])
plt.title("Egg domain")
plt.show()

# %% Run simulation
T_history, t_saved = bae.crank_nicolson_diffusion_2d(
    u_init,
    tmax,
    dt,
    dx,
    dy,
    unstructured_egg_domain=unstructured_egg_domain,
    nearest_neighbors=nearest_neighbors,
    egg_boundary_mesh_cells=egg_boundary_mesh_cells,
    water_temperature_celsius=WATER_TEMPERATURE_CELSIUS,
)


# %% Plotting


def convert_unstructured_array_to_structured(
    unstructured_arr: np.ndarray,
    map_from_mesh_cell_numbers_to_coords: dict[int, tuple[int, int]],
) -> np.ndarray:
    structured_arr = np.zeros((nx, ny))
    for cell_number, coords in map_from_mesh_cell_numbers_to_coords.items():
        structured_arr[coords] = unstructured_arr[cell_number]
    return structured_arr


def kelvin_to_celsius(T_kelvin: float | np.ndarray) -> float | np.ndarray:
    return T_kelvin - 273


def celsius_to_kelvin(T_celsius: float | np.ndarray) -> float | np.ndarray:
    return T_celsius + 273


T_history_structured = []
for u in T_history:
    T_history_structured.append(
        kelvin_to_celsius(
            convert_unstructured_array_to_structured(
                unstructured_arr=u,
                map_from_mesh_cell_numbers_to_coords=map_from_mesh_cell_numbers_to_coords,
            )
        )
    )

plot_times = [0, len(t_saved) // 4, len(t_saved) // 2, len(t_saved) - 1]

# Plot as 2D heat maps
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()

for i, time_idx in enumerate(plot_times):
    im = axes[i].imshow(
        T_history_structured[time_idx],
        origin="lower",
        extent=[0, Ly, 0, Lx],
        cmap="viridis",
        vmin=20,
        vmax=100,
    )
    axes[i].set_title(f"t = {t_saved[time_idx]:.4f}")
    axes[i].set_xlabel("X")
    axes[i].set_ylabel("Y")
    fig.colorbar(im, ax=axes[i])

    # add egg white and yolk
    xx = np.arange(start=0, stop=Lx, step=dx)
    white_yy = np.sqrt(bae.egg_curve_squared(a=Lx, b=B, x=xx))
    yolk_yy = np.sqrt(
        bae.yolk_curve_squared(yolk_radius=YOLK_RADIUS_METRES, Lx=Lx, x=xx)
    )
    axes[i].plot(white_yy, xx, c="white")
    axes[i].plot(yolk_yy, xx, c="black")

plt.tight_layout()
plt.show()


# %% Degree of cooking
# This code is perfectly fine. Move it to the boil_an_egg library!
#
# def log_white_A():
#     return np.log(4.85 * 10**60)
#
#
# def log_yolk_A():
#     return np.log(2.72 * 10**50)
#
#
# def white_Ea():
#     return 4.185 * 10**5
#
#
# def yolk_Ea():
#     return 3.443 * 10**5
#
#
# def log_A_egg(unstructured_egg_domain):
#     conditions = [
#         unstructured_egg_domain == 0,
#         unstructured_egg_domain == 1,
#         unstructured_egg_domain == 2,
#     ]
#     values = [0, log_white_A(), log_yolk_A()]
#     return np.select(condlist=conditions, choicelist=values)
#
#
# def Ea_egg(unstructured_egg_domain):
#     conditions = [
#         unstructured_egg_domain == 0,
#         unstructured_egg_domain == 1,
#         unstructured_egg_domain == 2,
#     ]
#     values = [0, white_Ea(), yolk_Ea()]
#     return np.select(condlist=conditions, choicelist=values)
#
#
# def R():
#     # J/(K*mol)
#     return 8.314
#
#
# nt = T_history.shape[0]
#
# degree_of_cooking_initial_condition = np.zeros_like(T_history[0])
# degree_of_cooking_history = [np.zeros(1)] * (nt - 1)
#
# degree_of_cooking_history[0] = degree_of_cooking_initial_condition
#
# Ea = Ea_egg(unstructured_egg_domain)
# log_A = log_A_egg(unstructured_egg_domain)
#
#
# # backwards Euler
#
# for timestep, t in enumerate(tqdm(range(1, nt))):
#     Ea = Ea_egg(unstructured_egg_domain)
#     log_A = log_A_egg(unstructured_egg_domain)
#
#     exponent = log_A - Ea / (R() * T_history[timestep])
#     b = degree_of_cooking_history[timestep - 1] + dt * np.exp(exponent)
#
#     A_matrix = bae.diags_array(1 / (1 + dt * np.exp(exponent)))
#
#     # Inverse of diagonal matrix is a diagonal matrix with elements 1/diagonal
#     degree_of_cooking_history[timestep] = A_matrix @ b
#
#
# # %% Plot as 2D heat maps
#
# degree_of_cooking_history_structured = []
# for X in degree_of_cooking_history:
#     degree_of_cooking_history_structured.append(
#         convert_unstructured_array_to_structured(
#             unstructured_arr=X,
#             map_from_mesh_cell_numbers_to_coords=map_from_mesh_cell_numbers_to_coords,
#         )
#     )
#
# timesteps_saved = len(degree_of_cooking_history)
# plot_times = [0, timesteps_saved // 4, timesteps_saved // 2, timesteps_saved - 1]
#
# fig, axes = plt.subplots(2, 2)
# axes = axes.flatten()
#
# for i, time_idx in enumerate(plot_times):
#     im = axes[i].imshow(
#         degree_of_cooking_history_structured[time_idx],
#         origin="lower",
#         extent=[0, Ly, 0, Lx],
#         cmap="viridis",
#         vmin=0,
#         vmax=1,
#     )
#     axes[i].set_title(f"t = {t_saved[time_idx]:.4f}")
#     axes[i].set_xlabel("X")
#     axes[i].set_ylabel("Y")
#     fig.colorbar(im, ax=axes[i])
#
#     # add egg white and yolk
#     xx = np.arange(start=0, stop=Lx, step=dx)
#     white_yy = np.sqrt(egg_curve_squared(a=Lx, b=B, x=xx))
#     yolk_yy = np.sqrt(yolk_curve_squared(yolk_radius=YOLK_RADIUS_METRES, Lx=Lx, x=xx))
#     axes[i].plot(white_yy, xx, c="white")
#     axes[i].plot(yolk_yy, xx, c="black")
#
# plt.tight_layout()
# plt.show()
#
# # %% Trials
#
# plt.imshow(
#     convert_unstructured_array_to_structured(Ea, map_from_mesh_cell_numbers_to_coords),
#     extent=[0, Ly, 0, Lx],
# )
# plt.show()
