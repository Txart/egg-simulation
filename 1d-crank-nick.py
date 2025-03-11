# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

# %%


def crank_nicolson_diffusion(u_init, tmax, dt, dx, C_func, k_func, boundary_conditions):
    """
    Solve the diffusion equation C(u) * du/dt = d/dx(k(u) * du/dx) using Crank-Nicolson method.

    Parameters:
    -----------
    u_init : array
        Initial condition for u
    tmax : float
        Maximum simulation time
    dt : float
        Time step size
    dx : float
        Spatial step size
    C_func : function
        Function defining the coefficient C(u)
    k_func : function
        Function defining the coefficient k(u)
    boundary_conditions : function
        Function that applies boundary conditions at each time step

    Returns:
    --------
    u_history : array
        Solution for all time steps
    t : array
        Time points
    """
    # Initialize
    nx = len(u_init)
    nt = int(tmax / dt) + 1

    # Storage for solution
    u_history = np.zeros((nt, nx))
    u_history[0] = u_init

    # Time points
    t = np.linspace(0, tmax, nt)

    # Coefficient for the finite difference scheme
    r = dt / (dx**2)

    # Main time loop
    for n in range(nt - 1):
        u = u_history[n].copy()

        # Update for nonlinearity: use simple fixed-point iteration
        max_iter = 10
        tolerance = 1e-6
        u_new = u.copy()

        for iteration in range(max_iter):
            # Calculate coefficients based on current approximation
            C_values = C_func(u)

            # Calculate k at cell interfaces (i+1/2 and i-1/2)
            k_values = k_func(u)
            k_half_plus = np.zeros_like(k_values)
            k_half_minus = np.zeros_like(k_values)

            # Harmonic mean for better handling of discontinuities
            for i in range(1, nx - 1):
                k_half_plus[i] = 0.5 * (k_values[i] + k_values[i + 1])
                k_half_minus[i] = 0.5 * (k_values[i] + k_values[i - 1])

            # Handle boundaries
            k_half_plus[0] = k_half_minus[1]  # Simple extrapolation
            k_half_minus[0] = k_values[0]
            k_half_plus[nx - 1] = k_values[nx - 1]
            k_half_minus[nx - 1] = k_half_plus[nx - 2]

            # Construct the tridiagonal system
            # Main diagonal
            main_diag = np.zeros(nx)
            for i in range(1, nx - 1):
                main_diag[i] = C_values[i] + 0.5 * r * (
                    k_half_plus[i] + k_half_minus[i]
                )

            # Set boundary conditions in the matrix
            main_diag[0] = 1.0  # Will be adjusted in boundary_conditions
            main_diag[nx - 1] = 1.0  # Will be adjusted in boundary_conditions

            # Upper diagonal
            upper_diag = np.zeros(nx - 1)
            for i in range(1, nx - 1):
                upper_diag[i] = -0.5 * r * k_half_plus[i]

            # Lower diagonal
            lower_diag = np.zeros(nx - 1)
            for i in range(1, nx - 1):
                lower_diag[i - 1] = -0.5 * r * k_half_minus[i]

            # Construct right-hand side
            b = np.zeros(nx)
            for i in range(1, nx - 1):
                explicit_term = (
                    0.5
                    * r
                    * (
                        k_half_plus[i] * (u[i + 1] - u[i])
                        - k_half_minus[i] * (u[i] - u[i - 1])
                    )
                )
                b[i] = C_values[i] * u[i] + explicit_term

            # Construct matrix A
            diagonals = [lower_diag, main_diag, upper_diag]
            offsets = [-1, 0, 1]
            A = (
                np.diag(lower_diag, k=-1)
                + np.diag(main_diag, k=0)
                + np.diag(upper_diag, k=1)
            )

            # Make sparse for efficiency
            A = sparse.csr_array(A)

            # Apply boundary conditions
            A, b = boundary_conditions(A, b, u, n, dt)

            # Solve system
            u_new_iter = spsolve(A, b)

            # Check convergence
            if np.max(np.abs(u_new_iter - u_new)) < tolerance:
                u_new = u_new_iter
                break

            u_new = u_new_iter

        # Store the result
        u_history[n + 1] = u_new

    return u_history, t


# Example usage
def dirichlet_boundary_conditions(A, b, u, n, dt):
    """
    Apply Dirichlet boundary conditions.

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

    Returns:
    --------
    A, b : updated matrix and right-hand side
    """
    nx = len(u)

    # Left boundary (u = 1)
    A[0, :] = 0
    A[0, 0] = 1
    b[0] = 100.0 + 273  # Kelvin

    # Right boundary (u = 0)
    A[nx - 1, :] = 0
    A[nx - 1, nx - 1] = 1
    b[nx - 1] = 20.0 + 273  # Kelvin

    return A, b


def constant_C(u):
    """Example: Constant C(u) = 1"""
    return np.ones_like(u)


def yolk_C(u):
    return 3120 * (1037.3 - 0.0023 * u**2 - 0.1386 * u)


def example_k(u):
    """Example: k(u) = 1 + u^2"""
    return 1.0 + u**2


def yolk_k(u):
    return 0.0008 * u + 0.395


def run_example():
    # Simulation parameters
    L = 0.08  # Domain length
    nx = 100  # Number of spatial points
    dx = L / (nx - 1)  # Spatial step size
    tmax = 60 * 10  # Maximum simulation time
    dt = 1.0  # Time step size

    # Initial condition (step function)
    x = np.linspace(0, L, nx)
    INITIAL_TEMPERATURE = 20 + 273  # Kelvin
    u_init = INITIAL_TEMPERATURE + np.ones(nx)

    # Run simulation
    u_history, t = crank_nicolson_diffusion(
        u_init,
        tmax,
        dt,
        dx,
        C_func=yolk_C,
        k_func=yolk_k,
        boundary_conditions=dirichlet_boundary_conditions,
    )

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot initial condition
    plt.subplot(2, 1, 1)
    plt.plot(x, u_init, "b-", label="Initial condition")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Initial Condition")
    plt.legend()

    # Plot solution at different times
    plt.subplot(2, 1, 2)
    num_lines = 5
    for i in range(num_lines):
        n = int((i / (num_lines - 1)) * (len(t) - 1))
        plt.plot(x, u_history[n], label=f"t = {t[n]:.3f}")

    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Solution Evolution")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_example()
