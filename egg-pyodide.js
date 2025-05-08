const output = document.getElementById("output");
const statusElement = document.getElementById("status");

function addToOutput(s) {
    output.value += ">>>" + "\n" + s + "\n";
}
function updateStatus(message) {
    statusElement.textContent = message;
}

output.value = "Initializing...\n";

// init Pyodide
async function main() {
    updateStatus("Loading Pyodide...");
    let pyodide = await loadPyodide();
    updateStatus("Loading micropip...");
    await pyodide.loadPackage("micropip");
    const micropip = pyodide.pyimport("micropip");
    updateStatus("Installing boil-an-egg package...");
    await micropip.install("boil-an-egg");
    output.value += "Ready!\n";
    updateStatus("Ready to run simulation");
    return pyodide;
}
let pyodideReadyPromise = main();

// Get parameter values from UI
function getParameters() {
    return {
        eggLength: parseFloat(document.getElementById("egg-length").value) /
            100, // Convert to meters
        yolkRadius: parseFloat(document.getElementById("yolk-radius").value) /
            100, // Convert to meters
        waterTemperature: parseFloat(
            document.getElementById("water-temperature").value,
        ),
        eggShape: parseFloat(document.getElementById("egg-shape").value),
        gridSize: parseInt(document.getElementById("grid-size").value),
    };
}

async function evaluatePython() {
    let pyodide = await pyodideReadyPromise;
    const runButton = document.getElementById("run-btn");

    try {
        runButton.disabled = true;
        updateStatus("Running simulation...");
        output.value = "Starting simulation with custom parameters...\n";
        // Get parameters from UI
        const params = getParameters();

        pyodide.runPython(`
import boil_an_egg.utils as bae
import numpy as np

EGG_LENGTH_METRES = ${params.eggLength}
YOLK_RADIUS_METRES = ${params.yolkRadius}
WATER_TEMPERATURE_CELSIUS = ${params.waterTemperature}
B = ${params.eggShape}  # Egg shape parameter
nx, ny = ${params.gridSize}, ${params.gridSize}  # Number of grid points

# Lx, Ly domain dimensions
Lx = EGG_LENGTH_METRES  # Domain dimensions = egg length
# y dimension depends on how wide the egg is
Ly = float(np.max(np.sqrt(bae.egg_curve_squared(a=Lx, b=B, x=np.linspace(0, Lx, nx)))))

dx = Lx / (nx - 1)  # Spatial step size in x
dy = Ly / (ny - 1)  # Spatial step size in y

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

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
T_init = (273 + 20) * np.ones(len(nearest_neighbors))

# Main loop
# Initialize
N = len(T_init)
tmax = 60 * 5 # Total simulation time in seconds
dt = 1 # timestep in seconds
nt = int(tmax / dt) + 1

n_saves = nt

# Storage for solution at saved times
T_history = np.zeros((n_saves, N))
t_saved = np.zeros(n_saves)

# Initial condition
T = T_init.copy()
T_history[0] = T
t_saved[0] = 0

# Main time loop
timestep=1 # substitute this with the for loop in the crank_nicolson function
save_idx = 1

# For the degree of cooking computation
degree_of_cooking_init = np.zeros_like(T_history[0]) # initial condition
degree_of_cooking = degree_of_cooking_init.copy()
degree_of_cooking_history = [np.zeros(1)] * (nt - 1)

degree_of_cooking_history[0] = degree_of_cooking

Ea = bae.Ea_egg(unstructured_egg_domain)
log_A = bae.log_A_egg(unstructured_egg_domain)
`);

        function compute_next_u() {
            pyodide.runPython(`
    T = bae.compute_next_u(
        u=T,
        dt=dt,
        dx=dx,
        dy=dy,
        unstructured_egg_domain=unstructured_egg_domain,
        nearest_neighbors=nearest_neighbors,
        egg_boundary_mesh_cells=egg_boundary_mesh_cells,
        water_temperature_celsius=WATER_TEMPERATURE_CELSIUS,
    )

    T_history[save_idx] = T
    t_saved[save_idx] = timestep * dt
    save_idx += 1


    T_celsius_structured = bae.kelvin_to_celsius(
        bae.convert_unstructured_array_to_structured(
            nx=nx,
            ny=ny,
            unstructured_arr=T,
            map_from_mesh_cell_numbers_to_coords=map_from_mesh_cell_numbers_to_coords,
        )
    )

    T_to_plot = bae.get_whole_egg(T_celsius_structured)
    `);
        }
        function compute_next_degree_of_cooking() {
            pyodide.runPython(`
    degree_of_cooking = bae.compute_next_degree_of_cooking(
        current_T=T,
        previous_degree_of_cooking=degree_of_cooking,
        dt=dt,
        Ea=Ea,
        log_A=log_A,
    )
    degree_of_cooking_to_plot = bae.get_whole_egg(
        bae.convert_unstructured_array_to_structured(
            nx=nx,
            ny=ny,
            unstructured_arr=degree_of_cooking,
            map_from_mesh_cell_numbers_to_coords=map_from_mesh_cell_numbers_to_coords,
        )
    )
`);
        }

        function heatmap_egg_quantity(
            egg_quantity,
            Lx,
            Ly,
            colorscale,
            zmin,
            zmax,
        ) {
            const data = [{
                z: egg_quantity,
                type: "heatmap",
                colorscale: colorscale,
                zmin: zmin,
                zmax: zmax,
                // Properly map array indices to the coordinate space
                x: Array.from(
                    { length: egg_quantity[0].length },
                    (_, i) => i * (2 * Ly) / (egg_quantity[0].length - 1),
                ),
                y: Array.from(
                    { length: egg_quantity.length },
                    (_, i) => i * Lx / (egg_quantity.length - 1),
                ),
            }];
            return data;
        }

        function plot_egg_temperature(pyodide) {
            const Ly = pyodide.runPython("Ly");
            const Lx = pyodide.runPython("Lx");
            const T_to_plot = pyodide.globals.get("T_to_plot").toJs();

            const data = heatmap_egg_quantity(
                T_to_plot,
                Lx,
                Ly,
                "Viridis",
                20,
                100,
            );
            Plotly.react("temperature_plot", data);
        }

        function plot_egg_doc(pyodide) {
            const Ly = pyodide.runPython("Ly");
            const Lx = pyodide.runPython("Lx");
            const doc_to_plot = pyodide.globals.get("degree_of_cooking_to_plot")
                .toJs();

            const data = heatmap_egg_quantity(
                doc_to_plot,
                Lx,
                Ly,
                "Viridis",
                0,
                1,
            );
            Plotly.react("doc_plot", data);
        }

        const nt = pyodide.globals.get("nt");
        const dt = pyodide.globals.get("dt");

        // Set up initial plots
        const T_init = pyodide.globals.get("T_init").toJs();
        const doc_init = pyodide.globals.get("degree_of_cooking_init").toJs();
        const Ly = pyodide.runPython("Ly");
        const Lx = pyodide.runPython("Lx");
        const data_T = [{
            z: T_init,
            type: "heatmap",
            colorscale: "Viridis",
            zmin: 20,
            zmax: 100,
            // Properly map array indices to the coordinate space
            x: Array.from(
                { length: T_init[0].length },
                (_, i) => i * (2 * Ly) / (T_init[0].length - 1),
            ),
            y: Array.from(
                { length: T_init.length },
                (_, i) => i * Lx / (T_init.length - 1),
            ),
        }];
        const data_doc = [{
            z: doc_init,
            type: "heatmap",
            colorscale: "Viridis",
            zmin: 20,
            zmax: 100,
            // Properly map array indices to the coordinate space
            x: Array.from(
                { length: T_init[0].length },
                (_, i) => i * (2 * Ly) / (T_init[0].length - 1),
            ),
            y: Array.from(
                { length: T_init.length },
                (_, i) => i * Lx / (T_init.length - 1),
            ),
        }];

        const layout = {
            title: "Heatmap",
            xaxis: {
                title: "X",
                range: [0, 2 * Ly],
                scaleanchor: "y",
                scaleratio: 1,
            },
            yaxis: {
                title: "Y",
                range: [0, Lx],
            },
        };

        Plotly.newPlot("temperature_plot", data_T, layout);
        Plotly.newPlot("doc_plot", data_doc, layout);

        async function runSimulation() {
            for (let step = 1; step < nt; step++) {
                // Compute temperatures
                compute_next_u();

                // Compute degree_of_cooking
                compute_next_degree_of_cooking();

                // Plots
                plot_egg_temperature(pyodide);
                plot_egg_doc(pyodide);

                // Add status indicator
                document.getElementById("status").textContent =
                    `Time elapsed (minutes): ${step * dt / 60}`;

                // Wait for the next animation frame to ensure the plot is rendered
                await new Promise((resolve) => setTimeout(resolve, 10));
            }
            document.getElementById("status").textContent =
                `Time elapsed (minutes): ${nt * dt / 60}. Simulation complete`;
        }
        await runSimulation();
    } catch (err) {
        addToOutput(err);
    } finally {
        runButton.disabled = false;
    }
}
