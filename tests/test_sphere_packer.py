from math import floor, pi
import time
import parakeet.freeze


def test_sphere_packer():
    # Set the volume size
    length_x = 30  # A
    length_y = 30
    length_z = 30
    volume = length_x * length_y * length_z  # A^3

    # Determine the number of waters to place
    avogadros_number = 6.02214086e23
    molar_mass_of_water = 18.01528  # grams / mole
    density_of_water = 940.0  # kg / m^3
    mass_of_water = (density_of_water * 1000) * (volume * 1e-10**3)  # g
    number_of_waters = int(
        floor((mass_of_water / molar_mass_of_water) * avogadros_number)
    )

    # Van der Waals radius of water
    van_der_waals_radius = 2.7 / 2.0  # A

    # Compute the total volume in the spheres
    volume_of_spheres = (4.0 / 3.0) * pi * van_der_waals_radius**3 * number_of_waters
    print("Fraction of volume filled: %.2f" % (100 * volume_of_spheres / volume))

    # Create the grid
    grid = (
        int(floor(length_z / (2 * van_der_waals_radius))),
        int(floor(length_y / (2 * van_der_waals_radius))),
        int(floor(length_x / (2 * van_der_waals_radius))),
    )

    # Compute the node length and density
    node_length = max([length_z / grid[0], length_y / grid[1], length_x / grid[2]])
    density = number_of_waters / volume

    # Create the packer
    packer = parakeet.freeze.SpherePacker(
        grid, node_length, density, van_der_waals_radius, max_iter=10
    )
    print(len(packer))

    # Extract all the data and compute the time taken
    start_time = time.time()
    coords = []
    for s in packer:
        for g in s:
            coords.extend(g)
        print(len(s))
    print("Time to compute: %f" % (time.time() - start_time))
    print(
        "Num unplaced samples: %d / %d"
        % (packer.num_unplaced_samples(), number_of_waters)
    )

    # Test overlaps
    min_distance_sq = (van_der_waals_radius * 2) ** 2
    for i in range(len(coords) - 1):
        ci = coords[i]
        for j in range(i + 1, len(coords)):
            cj = coords[j]
            d2 = (ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2 + (ci[2] - cj[2]) ** 2
            assert d2 >= min_distance_sq
