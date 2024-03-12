import numpy as np
import sys
sys.path.append('scripts')
from geosx_xml_object import SolversData, InternalMeshData, VTKMeshData, GeometryData, EventsData, ConstitutiveData, \
    InitialConditionData, BoundaryConditionData, RockParamData, FunctionData, PermPoroFunctionData, \
    XMLFileGEOSXGenerator, ElementRegionData, SourceTermData

def generate(darts_model, model_folder, physics_folder):
    # Set solver
    comp_names = darts_model.physics.property_containers[0].components_name
    nph = darts_model.physics.nph
    solvers_data = SolversData(num_comp=len(comp_names), num_phases=nph, trans_mult_exp=1)

    # Set mesh
    coords = np.array([[0., 500.], [0., 500.], [0., 20.]])
    npt = np.array([[25], [25], [1]])
    cell_sizes = np.diff(coords) / npt
    cell_block_names = ['block1']
    mesh_data = InternalMeshData(x_coords=coords[0], y_coords=coords[1], z_coords=coords[2],
                         nx=npt[0], ny=npt[1], nz=npt[2], cell_block_name=cell_block_names)

    # Define regions, should be consistent with mesh_data regions:
    region_name = 'Region1'
    material_name = 'rock'
    elements_regions_data = [ElementRegionData(name=region_name, region_name=cell_block_names,
                                                   material_list=material_name)]

    # Set initial conditions
    eps = darts_model.zero
    bar = 1.e+5
    p_init = darts_model.initial_values['pressure'] * bar
    t_init = 348.15
    z_init = []
    z_init_sum = np.sum([val for key, val in darts_model.initial_values.items() if key != 'pressure'])
    for comp in comp_names:
        if comp in darts_model.initial_values:
            z_init.append(darts_model.initial_values[comp])
        else:
            z_init.append(1. - z_init_sum)
    initial_condition_data = [InitialConditionData(pres_val=p_init,
                                                   temp_val=t_init,
                                                   comp_val=z_init,
                                                   comp_name=comp_names, region_name=region_name)]

    # Prescribe simulation and parameters related to output:
    day = 86400.0
    first_ts1 = day
    first_ts2 = 14 * day
    end1 = 7 * day
    end_final = 365 * day
    output_freq = 30 * day
    events_data = EventsData(max_time=end_final, out_name='outputs', out_freq=output_freq, out_tar_dir='/Outputs/vtkOutput',
                             solver_names=('solver_1', 'solver_2'), solver_dts=(first_ts1, first_ts2),
                             solver_end_time=(end1, end_final), solver_begin_time=(0, end1))

    # Set constitutives
    md = 0.9869e-15
    perm = [100 * md, 100 * md, 100 * md]
    porosity = 0.3
    compr = 1.e-9
    constitutive_data = [ConstitutiveData(ref_poro=porosity, ref_pres=p_init, compr=compr, perm=perm)]

    geometry_data = []
    boundary_condition_data = []
    # Set injector
    x_min = coords[:,0] - 0.01 * cell_sizes[:,0]
    x_max = 1.01 * cell_sizes[:,0]
    p_inj = 140. * bar
    geometry_data.append(GeometryData(x_min=x_min, x_max=x_max, name='inj1'))
    boundary_condition_data.append(BoundaryConditionData(pres_val=p_inj, temp_val=t_init,
                                                         comp_val=[1.- 2*eps, eps, eps],
                                                         comp_name=comp_names, region_name=region_name, source_name='inj1'))
    # Set producer
    x_min = coords[:,-1] - 1.01 * cell_sizes[:,0]
    x_max = coords[:,-1] + 0.01 * cell_sizes[:,0]
    p_prod = 50.0 * bar
    geometry_data.append(GeometryData(x_min=x_min, x_max=x_max, name='prd1'))
    boundary_condition_data.append(BoundaryConditionData(pres_val=p_prod, temp_val=t_init,
                                                         comp_val=[1.- 2*eps, eps, eps],
                                                         comp_name=comp_names, region_name=region_name, source_name='prd1'))

    rock_parameter_data = []
    # Define all rock related parameters (again, per region!):
    # rock_params_name = ['rockHeatCap', 'rockThermalConductivity', 'rockKineticRateFactor']
    # rock_params_fieldname = ['rockVolumetricHeatCapacity', 'rockThermalConductivity', 'rockKineticRateFactor']
    # rock_params_value = [2200, 181.44, 1.0]
    # rock_parameter_data = [RockParamData(rock_params_name=rock_params_name, rock_params_fieldname=rock_params_fieldname,
    #                                              rock_params_value=rock_params_value, region_name=region_name)]

    # Construct XML file and write to file:
    obl_file_name = '../.' + physics_folder + 'obl_table.txt'
    my_xml_file = XMLFileGEOSXGenerator(file_name=model_folder + 'input_file.xml', solvers_data=solvers_data, mesh_data=mesh_data,
                                        geometry_data=geometry_data, events_data=events_data, region_name=[region_name],
                                        constitutive_data=constitutive_data, elements_regions_data=elements_regions_data,
                                        initial_condition_data=initial_condition_data, rock_parameter_data=rock_parameter_data,
                                        boundary_condition_data=boundary_condition_data, obl_table_name=obl_file_name)
    my_xml_file.write_to_file()

    return