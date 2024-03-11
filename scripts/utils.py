import importlib.util
import sys
import os

def load_module_from_path(folder_path, module_name, entity_name=None):
    # Construct the full path to the module file
    module_path = os.path.join(folder_path, f"{module_name}.py")

    # Generate a spec for the module located at the given path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module named {module_name} from path {module_path}")

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module in its own namespace
    spec.loader.exec_module(module)

    # If an entity name is specified, import that entity
    if entity_name is not None:
        if hasattr(module, entity_name):
            entity = getattr(module, entity_name)
            return entity
        else:
            raise ImportError(f"Module '{module_name}' does not have an entity named '{entity_name}'")
    else:
        # If no specific entity is requested, return the whole module
        return module
