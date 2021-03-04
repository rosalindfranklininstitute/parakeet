import multem
import numpy
import pytest
import amplus.sample

# from matplotlib import pylab


def create_input_multislice(n_phonons, single_phonon_conf=False):

    # Initialise the input and system configuration
    input_multislice = multem.Input()

    # Set simulation experiment
    input_multislice.simulation_type = "HRTEM"

    # Electron-Specimen interaction model
    input_multislice.interaction_model = "Multislice"
    input_multislice.potential_type = "Lobato_0_12"

    # Potential slicing
    input_multislice.potential_slicing = "dz_Proj"

    # Electron-Phonon interaction model
    input_multislice.pn_model = "Still_Atom"
    input_multislice.pn_coh_contrib = 0
    input_multislice.pn_single_conf = False
    input_multislice.pn_nconf = 10
    input_multislice.pn_dim = 110
    input_multislice.pn_seed = 300_183

    # Specimen thickness
    input_multislice.thick_type = "Whole_Spec"

    # Illumination model
    input_multislice.illumination_model = "Partial_Coherent"
    input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

    input_multislice.nx = 1024
    input_multislice.ny = 1024

    input_multislice.obj_lens_m = 0
    input_multislice.obj_lens_c_10 = 15.836
    input_multislice.obj_lens_c_30 = 1e-03
    input_multislice.obj_lens_c_50 = 0.00
    input_multislice.obj_lens_c_12 = 0.0
    input_multislice.obj_lens_phi_12 = 0.0
    input_multislice.obj_lens_c_23 = 0.0
    input_multislice.obj_lens_phi_23 = 0.0
    input_multislice.obj_lens_inner_aper_ang = 0.0
    input_multislice.obj_lens_outer_aper_ang = 24.0
    input_multislice.obj_lens_zero_defocus_type = "Last"
    input_multislice.obj_lens_zero_defocus_plane = 0

    return input_multislice


def test_slice():

    filename = "temp.h5"
    box = (400, 400, 400)
    centre = (200, 200, 200)
    shape = { "type" : "cube", "cube" : { "length":400}}
    sample = amplus.sample.new(filename, box, centre, shape)
    amplus.sample.add_single_molecule(sample, "4v5d")

    # Create the system configuration
    system_conf = multem.SystemConfiguration()
    system_conf.precision = "float"
    system_conf.device = "device"

    # Create the input multislice configuration
    n_phonons = 50
    input_multislice = create_input_multislice(n_phonons, False)

    input_multislice.spec_atoms = list(sample.spec_atoms)
    input_multislice.spec_lx = sample.box_size[0]
    input_multislice.spec_ly = sample.box_size[1]
    input_multislice.spec_lz = sample.box_size[2]
    input_multislice.spec_dz = 3

    print("Standard")
    output = multem.simulate(system_conf, input_multislice)

    print("Subslicing")

    # Create the input multislice configuration
    n_slices = 4

    # Slice the sample
    subslices = list(
        multem.slice_spec_atoms(
            input_multislice.spec_atoms, input_multislice.spec_lz, n_slices
        )
    )

    # Do the slices simulation
    sliced_output = multem.simulate(system_conf, input_multislice, subslices)

    # Print the difference
    a = numpy.array(output.data[-1].m2psi_tot)
    b = numpy.array(sliced_output.data[-1].m2psi_tot)
    diff = numpy.max(numpy.abs(a - b))
    print(diff)
    assert diff == pytest.approx(0.0232, rel=1e-2)

    # fig, (ax1, ax2, ax3) = pylab.subplots(ncols=3)
    # ax1.imshow(a)
    # ax2.imshow(b)
    # ax3.imshow(a - b)
    # pylab.show()
