import importlib
import os


module = "swing_cutter"
module_file_name = f"{os.getcwd()}/{module}.py"
print(os.path.isfile(module_file_name))
importlib.import_module(".swing_cutter", ".")
