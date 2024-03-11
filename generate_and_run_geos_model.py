import numpy as np
import os
import subprocess
from scripts.utils import load_module_from_path
from scripts.geosx_obl_operators_object import OBLTableGenerator

def generate_geos_input_file(darts_folder, model_folder, regenerate_obl_table=True):
    # Load fluid physics from DARTS model
    Model = load_module_from_path(folder_path=physics_folder, module_name='model', entity_name='Model')
    darts_model = Model()

    # Generate OBL table and write in a file
    obl_table_file_name = physics_folder + 'obl_table.txt'
    if not os.path.exists(obl_table_file_name) or regenerate_obl_table:
        obl_table_gen = OBLTableGenerator(darts_model)
        obl_table_gen.generate_table()
        obl_table_gen.write_table_to_file(filename=obl_table_file_name)

    # Generate GEOS input *.xml
    generate = load_module_from_path(folder_path=model_folder, module_name='generate_xml', entity_name='generate')
    generate(darts_model=darts_model, model_folder=model_folder, physics_folder=physics_folder)

# Generate OBL tables & GEOS input file
physics_folder = './darts_physics/_2ph_comp/'
model_folder = './geos_models/square_two_wells/'
generate_geos_input_file(darts_folder=physics_folder, model_folder=model_folder, regenerate_obl_table=True)

# Run GEOS
os.chdir(model_folder)
geos_path = './../../../../geos/build-your-platform-release/bin/geosx'
run_geos = [geos_path, '-i', 'input_file.xml']
result = subprocess.run(run_geos, capture_output=False, text=True)






