# This was cowritten by Claude AI and me

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import cm
from tqdm import tqdm


# %%
def crank_nicolson_diffusion_2d(
    u_init,
    tmax,
    dt,
    dx,
    dy,
    egg_domain,
    egg_to_equation_system_map,
):
    """
    Solve the 2D diffusion equation C(u) * du/dt = d/dx(k(u) * du/dx) + d/dy(k(u) * du/dy) using Crank-Nicolson method.

    Parameters:
    -----------
    u_init : 2D array
        Initial condition for u
    tmax : float
        Maximum simulation time
    dt : float
        Time step size
    dx : float
        Spatial step size in x direction
    dy : float
        Spatial step size in y direction
    nearest_neighbors:dict[int, dict[str,int|None]]
        Nearest neighbor mesh cell numbers for each mesh cell
    egg_boundary_mesh_cells: list[int]
        Mesh cell numbers corresponding to egg boundaries

    Returns:
    --------
    u_history : array
        Solution for selected time steps
    t_saved : array
        Saved time points
    """
    # Initialize
    N = len(u_init)
    nt = int(tmax / dt) + 1

    # We'll save fewer time steps to save memory
    save_interval = max(1, nt // 20)  # Save approximately 20 snapshots
    n_saves = nt // save_interval + 1

    # Storage for solution at saved times
    u_history = np.zeros((n_saves, N))
    t_saved = np.zeros(n_saves)

    # Initial condition
    u = u_init.copy()
    u_history[0] = u
    t_saved[0] = 0

    # Main time loop
    save_idx = 1
    for timestep in tqdm(range(1, nt)):
        u = compute_next_u(
            u=u,
            timestep=timestep,
            dt=dt,
            dx=dx,
            dy=dy,
            egg_domain=egg_domain,
            egg_to_equation_system_map=egg_to_equation_system_map,
        )

        # Save at specified intervals
        if timestep % save_interval == 0:
            u_history[save_idx] = u
            t_saved[save_idx] = timestep * dt
            save_idx += 1

    # Ensure the final state is saved
    if (nt - 1) % save_interval != 0:
        u_history[save_idx] = u
        t_saved[save_idx] = (nt - 1) * dt
        save_idx += 1

    return u_history[:save_idx], t_saved[:save_idx]


def compute_next_u(
    u,
    timestep,
    dt,
    dx,
    dy,
    egg_domain,
    egg_to_equation_system_map,
):
    # Update for nonlinearity: use simple fixed-point iteration
    max_iter = 10
    tolerance = 1e-6
    u_new = u.copy()

    for iteration in range(max_iter):
        A, b = build_matrix_and_b_equations(
            u=u,
            dt=dt,
            dx=dx,
            dy=dy,
            unstructured_egg_domain=unstructured_egg_domain,
            nearest_neighbors=nearest_neighbors,
            egg_boundary_mesh_cells=egg_boundary_mesh_cells,
        )

        # Apply boundary conditions
        A, b = dirichlet_boundary_conditions(
            A=A,
            b=b,
            dt=dt,
            water_bath_temperature_BC=WATER_TEMPERATURE_CELSIUS + 273,
            egg_boundary_mesh_cells=egg_boundary_mesh_cells,
        )

        # Solve system
        u_new_iter = spsolve(A, b)

        # Check convergence
        if np.max(np.abs(u_new_iter - u_new)) < tolerance:
            u_new = u_new_iter
            break

        u_new = u_new_iter

        return u_new


def build_matrix_and_b_equations(
    u,
    dt,
    dx,
    dy,
    unstructured_egg_domain,
    nearest_neighbors,
    egg_boundary_mesh_cells,
):
    N = len(u)

    interior_mesh_cell_numbers = tuple(set(range(N)) - set(egg_boundary_mesh_cells))

    rx = dt / (2 * dx**2)
    ry = dt / (2 * dy**2)

    # Calculate coefficients based on current approximation
    C_values = C_egg(u, unstructured_egg_domain)
    k_values = k_egg(u, unstructured_egg_domain)

    # Calculate k at cell interfaces (i+1/2, i-1/2, j+1/2, j-1/2)
    # For simplicity, we use arithmetic mean
    k_right = np.zeros_like(k_values)
    k_left = np.zeros_like(k_values)
    k_up = np.zeros_like(k_values)
    k_down = np.zeros_like(k_values)

    # Interior points
    for cell_number in interior_mesh_cell_numbers:
        k_right[cell_number] = 0.5 * (
            k_values[cell_number] + k_values[nearest_neighbors[cell_number]["right"]]
        )
        k_left[cell_number] = 0.5 * (
            k_values[cell_number] + k_values[nearest_neighbors[cell_number]["left"]]
        )
        k_up[cell_number] = 0.5 * (
            k_values[cell_number] + k_values[nearest_neighbors[cell_number]["up"]]
        )
        k_down[cell_number] = 0.5 * (
            k_values[cell_number] + k_values[nearest_neighbors[cell_number]["down"]]
        )

    # Initialize A matrix quantities
    data, row_ind, col_ind = [], [], []

    # Initialize b
    b = np.zeros(N)

    for cell_number in interior_mesh_cell_numbers:
        # Coefficients for implicit part
        # Center point
        center_coeff = (
            C_values[cell_number]
            + rx * (k_right[cell_number] + k_left[cell_number])
            + ry * (k_up[cell_number] + k_down[cell_number])
        )

        # Neighboring points
        left_coeff = -rx * k_left[cell_number]
        right_coeff = -rx * k_right[cell_number]
        down_coeff = -ry * k_down[cell_number]
        up_coeff = -ry * k_up[cell_number]

        # Add center point
        row_ind.append(cell_number)
        col_ind.append(cell_number)
        data.append(center_coeff)

        # Add left point
        row_ind.append(cell_number)
        col_ind.append(nearest_neighbors[cell_number]["left"])
        data.append(left_coeff)

        # Add right point
        row_ind.append(cell_number)
        col_ind.append(nearest_neighbors[cell_number]["right"])
        data.append(right_coeff)

        # Add down point
        row_ind.append(cell_number)
        col_ind.append(nearest_neighbors[cell_number]["down"])
        data.append(down_coeff)

        # Add up point
        row_ind.append(cell_number)
        col_ind.append(nearest_neighbors[cell_number]["up"])
        data.append(up_coeff)

        # Right-hand side (explicit part)
        explicit_term_x = rx * (
            k_right[cell_number]
            * (u[nearest_neighbors[cell_number]["right"]] - u[cell_number])
            - k_left[cell_number]
            * (u[cell_number] - u[nearest_neighbors[cell_number]["left"]])
        )
        explicit_term_y = ry * (
            k_up[cell_number]
            * (u[nearest_neighbors[cell_number]["up"]] - u[cell_number])
            - k_down[cell_number]
            * (u[cell_number] - u[nearest_neighbors[cell_number]["down"]])
        )

        b[cell_number] = (
            C_values[cell_number] * u[cell_number] + explicit_term_x + explicit_term_y
        )

    # Construct sparse matrix
    A = csr_matrix((data, (row_ind, col_ind)), shape=(N, N))

    return A, b


def dirichlet_boundary_conditions(
    A, b, dt, water_bath_temperature_BC: float, egg_boundary_mesh_cells: tuple[int]
):
    """
    Apply boundary conditions. There are two types of BC in our problem:
    - true egg boundary: dirichlet boundary conditions at water bath temperature
    - mirror egg boundary: these arise because an egg has a symmetry axis and we only compute half an egg.
    These are automatically handled by the general algorithm to assign A and b values.

    Thus, here we only apply the true egg BCs.

    Parameters:
    -----------
    A : sparse matrix
        System matrix
    b : array
        Right-hand side
    u : array
        Current solution
    n : int
        Current time step
    dt : float
        Time step size
    nx, ny : int
        Grid dimensions
    egg_boundary_mesh_cells
        Mesh cell numbers corresponding to egg boundaries

    Returns:
    --------
    A, b : updated matrix and right-hand side
    """

    # Bottom and top boundaries (y = 0 and y = Ly)
    for boundary_cell_number in egg_boundary_mesh_cells:
        A[boundary_cell_number] = 0
        A[boundary_cell_number, boundary_cell_number] = 1
        b[boundary_cell_number] = water_bath_temperature_BC

    return A, b


def yolk_k(u):
    return 0.0008 * u + 0.395


def white_k(u):
    return 0.0013 * u + 0.5125


def yolk_C(u):
    return 3120 * (1037.3 - 0.0023 * u**2 - 0.1386 * u)


def white_C(u):
    return 3800


def k_egg(u, unstructured_egg_domain):
    conditions = [
        unstructured_egg_domain == 0,
        unstructured_egg_domain == 1,
        unstructured_egg_domain == 2,
    ]
    values = [0, white_k(u), yolk_k(u)]
    return np.select(condlist=conditions, choicelist=values)


def C_egg(u, unstructured_egg_domain):
    conditions = [
        unstructured_egg_domain == 0,
        unstructured_egg_domain == 1,
        unstructured_egg_domain == 2,
    ]
    values = [0, white_C(u), yolk_C(u)]
    return np.select(condlist=conditions, choicelist=values)


def is_point_outside_egg(i, j, egg_domain):
    # i and j are array indices

    # egg_domain = 0 is outside of egg
    return egg_domain[i, j] < 0.5


def egg_curve_squared(a: float, b: float, x: float | np.ndarray) -> float | np.ndarray:
    return x * 0.5 * ((a - b) - 2 * x + np.sqrt(4 * b * x + (a - b) ** 2))


def create_egg_domain(
    nx, ny, Lx, Ly, yolk_radius_metres, B_EGG_SHAPE_PARAM
) -> np.ndarray:
    # 0 = outside
    # 1 = white
    # 2 = yolk

    egg_domain = np.zeros(shape=(nx, ny))
    xx = np.arange(start=0, stop=Lx, step=Lx / nx)
    yy = np.arange(start=0, stop=Ly, step=Ly / ny)

    for i, _ in enumerate(egg_domain):
        for j, _ in enumerate(egg_domain[i]):
            x = xx[i]
            y = yy[j]

            if y**2 <= egg_curve_squared(a=Lx, b=B_EGG_SHAPE_PARAM, x=x):
                egg_domain[i, j] = 1

            if (x - 2 * Lx / 3) ** 2 + y**2 <= yolk_radius_metres**2:
                egg_domain[i, j] = 2

    return egg_domain


def compute_egg_to_equation_system_map(nx, ny, egg_domain):
    egg_to_equation_system_map = -np.ones((nx, ny), dtype=np.int16)
    cell_number = 0
    for i in range(nx):
        for j in range(ny):
            if not is_point_outside_egg(i, j, egg_domain):
                egg_to_equation_system_map[i, j] = cell_number
                cell_number += 1

    return egg_to_equation_system_map


def nearest_neighbors_of_single_cell(
    coords_in_grid: tuple[int, int],
) -> dict[str, int | None]:
    # Mesh boundaries
    if coords_in_grid[1] == 0:  # mirror BC
        left = egg_to_equation_system_map[
            coords_in_grid[0], coords_in_grid[1] + 1
        ].item()
    else:
        # General
        left = egg_to_equation_system_map[
            coords_in_grid[0], coords_in_grid[1] - 1
        ].item()

    if coords_in_grid[1] == ny - 1:  # domain diri BC
        right = None
    else:
        right = egg_to_equation_system_map[
            coords_in_grid[0], coords_in_grid[1] + 1
        ].item()

    if coords_in_grid[0] == 0:  # domain diri BC
        up = None
    else:
        up = egg_to_equation_system_map[coords_in_grid[0] - 1, coords_in_grid[1]].item()

    if coords_in_grid[0] == nx - 1:  # domain diri BC
        down = None
    else:
        down = egg_to_equation_system_map[
            coords_in_grid[0] + 1, coords_in_grid[1]
        ].item()

    # Egg boundaries
    if right == -1:
        right = None
    if up == -1:
        up = None
    if down == -1:
        down = None
    if left == -1:
        left = None

    return {"left": left, "right": right, "up": up, "down": down}


def map_mesh_cell_numbers_to_coords(
    egg_to_equation_system_map: np.ndarray,
) -> dict[int, tuple[int, int]]:
    n_unstructured_mesh_cells = egg_to_equation_system_map.max()
    mesh_cell_number_coords = {}

    for cell_number in range(n_unstructured_mesh_cells + 1):
        coords_in_grid = np.where(egg_to_equation_system_map == cell_number)

        mesh_cell_number_coords[cell_number] = (
            coords_in_grid[0].item(),
            coords_in_grid[1].item(),
        )
    return mesh_cell_number_coords


def get_nearest_neighbors(
    map_from_mesh_cell_numbers_to_coords: dict[int, tuple[int, int]],
) -> dict[int, dict[str, int | None]]:
    nearest_neighbors = {}
    for cell_number, cell_coords in map_from_mesh_cell_numbers_to_coords.items():
        nearest_neighbors[cell_number] = nearest_neighbors_of_single_cell(cell_coords)

    return nearest_neighbors


def get_egg_boundary_mesh_cells(nearest_neighbors):
    # Mesh cells which are egg boundaries
    return tuple(
        [
            cell_number
            for cell_number, nn in nearest_neighbors.items()
            if None in nn.values()
        ]
    )


def invert_dictionary(dictionary, are_values_unique=True):
    """Invert an input dictionary, with the option to keep all non-unique values

    Args:
        dictionary (dict): Dictionary to invert.
        are_values_unique (bool, optional): If True, then the inverted dictionary will only have one value per key. If False, it creates a list of all the values. Defaults to True.

    Returns:
        dict: Inverted dictionary.
    """
    if are_values_unique:
        inv_dict = {v: k for k, v in dictionary.items()}

    else:  # more than one value per key; keep all values
        inv_dict = {}
        for k, v in dictionary.items():
            inv_dict[v] = inv_dict.get(v, []) + [k]

    return inv_dict


def create_unstructured_array_from_structured_array(
    structured_array: np.ndarray, map_from_mesh_cell_numbers_to_coords
) -> np.ndarray:
    unstructured_array = np.zeros(len(map_from_mesh_cell_numbers_to_coords))
    for num, coords in map_from_mesh_cell_numbers_to_coords.items():
        unstructured_array[num] = structured_array[coords]

    return unstructured_array


# %%
# Simulation parameters
EGG_LENGTH_METRES = 8 / 100
YOLK_RADIUS_METRES = 1.8 / 100
WATER_TEMPERATURE_CELSIUS = 100
B = 0.09  # Egg shape parameter
nx, ny = 10, 10  # Number of grid points

# Lx, Ly domain dimensions
Lx = EGG_LENGTH_METRES  # Domain dimensions = egg length
# y dimension depends on how wide the egg is
Ly = float(np.max(np.sqrt(egg_curve_squared(a=Lx, b=B, x=np.linspace(0, Lx, nx)))))

dx = Lx / (nx - 1)  # Spatial step size in x
dy = Ly / (ny - 1)  # Spatial step size in y
tmax = 60 * 5  # Maximum simulation time
dt = 1  # Time step size

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Separate white and yolk
egg_domain = create_egg_domain(
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
egg_to_equation_system_map = compute_egg_to_equation_system_map(
    nx=nx, ny=ny, egg_domain=egg_domain
)
map_from_mesh_cell_numbers_to_coords = map_mesh_cell_numbers_to_coords(
    egg_to_equation_system_map
)

map_from_coords_to_mesh_cell_numbers = invert_dictionary(
    dictionary=map_from_mesh_cell_numbers_to_coords, are_values_unique=True
)

unstructured_egg_domain = create_unstructured_array_from_structured_array(
    structured_array=egg_domain,
    map_from_mesh_cell_numbers_to_coords=map_from_mesh_cell_numbers_to_coords,
)

nearest_neighbors = get_nearest_neighbors(
    map_from_mesh_cell_numbers_to_coords=map_from_mesh_cell_numbers_to_coords
)
egg_boundary_mesh_cells = get_egg_boundary_mesh_cells(
    nearest_neighbors=nearest_neighbors
)

# Initial condition
u_init = (273 + 20) * np.ones(len(nearest_neighbors))

# Plot egg domain
plt.figure()
plt.imshow(egg_domain, extent=[0, Ly, 0, Lx])
plt.title("Egg domain")
plt.show()

# Run simulation
u_history, t_saved = crank_nicolson_diffusion_2d(
    u_init,
    tmax,
    dt,
    dx,
    dy,
    egg_domain=egg_domain,
    egg_to_equation_system_map=egg_to_equation_system_map,
)

# Plot results
plot_times = [0, len(t_saved) // 4, len(t_saved) // 2, len(t_saved) - 1]

fig = plt.figure(figsize=(16, 12))
for i, time_idx in enumerate(plot_times):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    surf = ax.plot_surface(
        X, Y, u_history[time_idx], cmap=cm.viridis, linewidth=0, antialiased=True
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("u")
    ax.set_title(f"t = {t_saved[time_idx]:.4f}")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# Plot as 2D heat maps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, time_idx in enumerate(plot_times):
    im = axes[i].imshow(
        u_history[time_idx],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="viridis",
        vmin=273,
        vmax=273 + 100,
    )
    axes[i].set_title(f"t = {t_saved[time_idx]:.4f}")
    axes[i].set_xlabel("X")
    axes[i].set_ylabel("Y")
    fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()

# %% Trials

A, b = build_matrix_and_b_equations(
    u=u_init,
    dt=dt,
    dx=dx,
    dy=dy,
    unstructured_egg_domain=unstructured_egg_domain,
    nearest_neighbors=nearest_neighbors,
    egg_boundary_mesh_cells=egg_boundary_mesh_cells,
)

# Apply boundary conditions
A, b = dirichlet_boundary_conditions(
    A=A,
    b=b,
    dt=dt,
    water_bath_temperature_BC=WATER_TEMPERATURE_CELSIUS + 273,
    egg_boundary_mesh_cells=egg_boundary_mesh_cells,
)
