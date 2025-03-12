- Egg shaped curve from here: https://nyjp07.com/index_egg_E.html

$$
(x^2 + y^2)^2 = a*x^3 + (a-b)*x*y^2

- Only consider half of an egg (symmetry). In my case, because the eggo is lying down, y>0.

# TODOs
- get better fit of domain to egg shape. Right now, N_Y is not computed according to the shape, so a lot of space is wasted in the array.

# IDEAS
- Make a transformation from cartesian coords to "ovoid" coords or radial coords and transform the PDE to solve. Might make the solution computationally less expensive (there are no "dead" cells), and it is more elegant overall. We like elegance.
