- Egg shaped curve from here: https://nyjp07.com/index_egg_E.html

$$
(x^2 + y^2)^2 = a*x^3 + (a-b)*x*y^2

- Only consider half of an egg (symmetry). In my case, because the eggo is lying down, y>0.

# TODOs
- Move towards unstructured grid using egg_to_equation_system_map.
- With the unstructured mesh, the matrix is not that sparse any more. Do we need sparse matrices??
- Create a way to plot unstructured mesh
# IDEAS
- Make a transformation from cartesian coords to "ovoid" coords or radial coords and transform the PDE to solve. Might make the solution computationally less expensive (there are no "dead" cells), and it is more elegant overall. We like elegance.
