import numpy as np
import sys
sys.path.append('scripts')
from geosx_xml_object import SolversData, VTKMeshData, GeometryData, EventsData, ConstitutiveData, \
    InitialConditionData, BoundaryConditionData, RockParamData, FunctionData, PermPoroFunctionData, \
    XMLFileGEOSXGenerator, ElementRegionData, SourceTermData

def generate(darts_model, model_folder, physics_folder):
    domain_sizes = np.array([8400, 5000, 1200])
    n_cells = np.array([50, 50, 50])
    cell_sizes = domain_sizes / n_cells
    p_init_well1 = 3e7

    # Set solver
    comp_names = darts_model.physics.property_containers[0].components_name
    nph = darts_model.physics.nph
    solvers_data = SolversData(num_comp=len(comp_names), num_phases=nph,
                               trans_mult_exp=1, solver_type='iterative')

    # Set mesh
    mesh_data = VTKMeshData(name_mesh='mesh1', name_vtu_file='spe11c_structured.vtu')

    # Set constitutives
    compr = 1.e-9
    unit_perm = np.array([1., 1., 1.])
    material_lists = ['seal1', 'sand2', 'sand3', 'sand4', 'sand5', 'sand6', 'impermeable7']
    constitutive_data = []
    for material in material_lists:
        if material == 'seal1':
            perm = 1.e-16
            poro = 0.1
        elif material == 'sand2':
            perm = 1.e-13
            poro = 0.2
        elif material == 'sand3':
            perm = 2.e-13
            poro = 0.2
        elif material == 'sand4':
            perm = 5.e-13
            poro = 0.2
        elif material == 'sand5':
            perm = 1.e-12
            poro = 0.25
        elif material == 'sand6':
            perm = 2.e-12
            poro = 0.35
        elif material == 'impermeable7':
            perm = 1.e-25
            poro = 1.e-5

        constitutive_data.append(ConstitutiveData(name=material, ref_poro=poro, ref_pres=p_init_well1, compr=compr, perm=perm * unit_perm))

    # Define regions, should be consistent with mesh_data regions:
    region_names = ['Facies1', 'Facies2', 'Facies3', 'Facies4', 'Facies5', 'Facies6', 'Facies7']
    regions = [['1_hexahedra'], ['2_hexahedra'], ['3_hexahedra'], ['4_hexahedra'], ['5_hexahedra'], ['6_hexahedra'], ['7_hexahedra']]
    elements_regions_data = []
    for i in range(len(region_names)):
        elements_regions_data.append(ElementRegionData(name=region_names[i], region_name=regions[i],
                                                       material_list=material_lists[i]))

    # Prescribe simulation and parameters related to output:
    day = 86400.0
    first_ts1 = 0.2 * day
    first_ts2 = 2 * day
    end1 = 2 * day
    end_final = 365 * 1000 * day # 1000 years
    out_freq = 10 * day
    events_data = EventsData(max_time=end_final, out_name='outputs', out_freq=out_freq, out_tar_dir='/Outputs/vtkOutput',
                             solver_names=('solver_1', 'solver_2'), solver_dts=(first_ts1, first_ts2),
                             solver_end_time=(end1, end_final), solver_begin_time=(0, end1))

    # Set initial conditions
    eps = darts_model.zero
    initial_condition_data = []
    t_init = 348.15
    z_init = [1 - 10*eps, 10*eps]
    for i, region in enumerate(region_names):
        # initial_condition_data.append(InitialConditionData(pres_val=1e7, temp_val=348.15, comp_val=[1 - eps, eps],
        #                                                    comp_name=comp_name, region_name=region))
        initial_condition_data.append(InitialConditionData(pres_val=p_init_well1, temp_val=t_init,
                                                           comp_val=z_init,
                                                           comp_name=comp_names, region_name=region))

    # Set wells
    geometry_data = list()
    x_min = np.array([2700.0, 1000.0, 300.0])
    x_max = np.array([2700.0, 4000.0, 300.0])
    geometry_data.append(GeometryData(x_min=x_min - cell_sizes, x_max=x_max + cell_sizes, name='inj1'))
    # geometry_data.append(GeometryData(x_min=(5100.0, 1000.0, 700.0),
    #                                  x_max=(5100.0, 4000.0, 700.0),
    #                                  name='source2'))
    # source_term_data = []
    # kmol_gas_per_sec = (100 / 44.01) * 0.2 / (60 * 60 * 24)
    # source_term_data.append(SourceTermData(name='inj1_co2', region_name='rock', component=0,
    #                                        scale=kmol_gas_per_sec, source_name='inj1'))

    boundary_condition_data = []
    for i, region in enumerate(region_names):
        boundary_condition_data.append(BoundaryConditionData(pres_val=4.e7, temp_val=348.15,
                                                             comp_val=[10 * eps, 1 - 10 * eps],
                                                             comp_name=comp_names, region_name=region, source_name='inj1'))

    # Define all rock related parameters (again, per region!):
    rock_params_name = ['rockHeatCap', 'rockThermalConductivity', 'rockKineticRateFactor']
    rock_params_fieldname = ['rockVolumetricHeatCapacity', 'rockThermalConductivity', 'rockKineticRateFactor']
    rock_params_value = [2200, 181.44, 1.0]
    rock_parameter_data = []
    for i, region in enumerate(region_names):
        rock_parameter_data.append(RockParamData(rock_params_name=rock_params_name, rock_params_fieldname=rock_params_fieldname,
                                                 rock_params_value=rock_params_value, region_name=region))

    # Construct XML file and write to file:
    obl_file_name = '../.' + physics_folder + 'obl_table.txt'
    my_xml_file = XMLFileGEOSXGenerator(file_name=model_folder + 'input_file.xml', solvers_data=solvers_data, mesh_data=mesh_data,
                                        geometry_data=geometry_data, events_data=events_data, region_name=region_names,
                                        constitutive_data=constitutive_data, elements_regions_data=elements_regions_data,
                                        initial_condition_data=initial_condition_data, rock_parameter_data=rock_parameter_data,
                                        boundary_condition_data=boundary_condition_data, obl_table_name=obl_file_name)
    my_xml_file.write_to_file()

    return




# # Generate XML file:
# comp_name = ['Comp_CO2', 'Comp_Ions', 'Comp_H2O', 'Comp_CaCO3']
# solvers_data = SolversData(num_comp=len(comp_name), num_phases=2)
#
# mesh_data = MeshData(x_coords=(0, 1000), y_coords=(0, 1), z_coords=(0, 1),
#                      nx=(1000), ny=(1), nz=(1),
#                      cell_block_name=['matrix00'])
#
# # Define geometries for boundary conditions:
# geometry_data = list()
# geometry_data.append(GeometryData(x_min=(-0.01, -0.01, -0.01),
#                                   x_max=(1.01, 1.01, 1.01),
#                                   name='source_gas'))
# geometry_data.append(GeometryData(x_min=(998.99, -0.01, -0.01),
#                                   x_max=(1000.01, 1.01, 1.01),
#                                   name='sink'))
#
# # Prescribe simulation and output related parameters:
# end_first_tz = 1e4
# end_final = 8.64e7
# events_data = EventsData(max_time=end_final, out_name='outputs', out_freq=1e6, out_tar_dir='/Outputs/vtkOutput',
#                          solver_names=('solver_1', 'solver_2'), solver_dts=(1e3, 5e4), solver_end_time=(end_first_tz, end_final),
#                          solver_begin_time=(0, end_first_tz))
#
# constitutive_data = ConstitutiveData(perm=(3.7e-12, 3.7e-12, 3.7e-12))
#
# # Define regions, should be consistent with mesh_data regions:
# elements_regions_data = list()
# elements_regions_data.append(ElementRegionData(name='matrix',
#                                                region_name=['matrix00']))
#
# # Define parameters for initial and boundary conditions (for each region!):
# initial_condition_data = list()
# initial_condition_data.append(InitialConditionData(pres_val=1e7, temp_val=348.15,
#                                                    comp_val=[1e-10, 0.1499999, 0.1499999, 0.7],
#                                                    comp_name=comp_name, region_name='matrix'))
#
# boundary_condition_data = list()
# boundary_condition_data.append(BoundaryConditionData(pres_val=9.5e6, temp_val=348.15,
#                                                      comp_val=[1e-10, 0.1499999, 0.1499999, 0.7],
#                                                      comp_name=comp_name, region_name='matrix', source_name='sink'))
#
# source_term_data = list()
# kmol_gas_per_sec = (100 / 44.01) * 0.2 / (60 * 60 * 24)
# source_term_data.append(SourceTermData(name='sourceGas', region_name='matrix',
#                                        component=0, scale=-kmol_gas_per_sec,
#                                        source_name='source_gas'))
#
# # Define all rock related parameters (again, per region!):
# rock_params_name = ['rockHeatCap', 'rockThermalConductivity', 'rockKineticRateFactor']
# rock_params_fieldname = ['rockVolumetricHeatCapacity', 'rockThermalConductivity', 'rockKineticRateFactor']
# rock_params_value = [2200, 181.44, 1.0]
# rock_parameter_data = list()
# rock_parameter_data.append(RockParamData(rock_params_name=rock_params_name, rock_params_fieldname=rock_params_fieldname,
#                                          rock_params_value=rock_params_value, region_name='matrix'))
#
# # Construct XML file and write to file:
# my_xml_file = XMLFileGEOSXGenerator(file_name='benchmark_1D_with_source.xml', solvers_data=solvers_data, mesh_data=mesh_data,
#                                     geometry_data=geometry_data, events_data=events_data,
#                                     constitutive_data=constitutive_data, elements_regions_data=elements_regions_data,
#                                     boundary_condition_data=boundary_condition_data,
#                                     table_function_data=None, rock_parameter_data=rock_parameter_data,
#                                     initial_condition_data=initial_condition_data,
#                                     perm_poro_functions=None, source_term_data=source_term_data,
#                                     obl_table_name='benchmark_operators.txt', region_name=['matrix'])
# my_xml_file.write_to_file()
