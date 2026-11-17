# save_load_example
import sys
from pathlib import Path

# Add the repository root to Python's import path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Get the path to the directory we're saving/loading data from
EXAMPLE_DATA = Path(__file__).resolve().parent / "example_data"

import numpy as np
import matplotlib.pyplot as plt
import MainLoopStreamline as ml
from OrbitPlotter import plotvalues

RUN_NEW_SIM = True

if RUN_NEW_SIM:
    # Gotta do a big ol example inspiral, lets do something with 1e-4?
    # This took about 2 minutes to run on the author's laptop
    orbit = ml.EMRIGenerator(
        a=0.8, 
        mu=1e-4,
        params=[6, 0.4, np.pi/4],
        err_target=1e-12,
        label="example generic orbit")

    # This saves the resulting data as an hdf5 file in a subdirectory called 'example_data' 
    #   inside the 'examples' directory. 
    ml.save_emri_data(orbit, filename="example_inspiral", folder=EXAMPLE_DATA)

# It also creates an 'index.json' file that tracks useful information about the saved data.
# We can load the index and retrieve that information without reloading everything.
index = ml.load_index(folder=EXAMPLE_DATA)
for filename, metadata in index.items():
    print("----------")
    print(f"Filename: {filename}")
    print(f"    Internal Label: {metadata.get('Label', 'Unknown')}")
    print(f"    Date/Time created: {metadata.get('Created', 'Unknown')}")
    print(f"    Ended in plunge: {metadata.get('Plunged', 'Unknown')}")
print("----------")

# Now we can load up our EMRI data- this one is identical to the one above.
loaded_orbit = ml.load_emri_data(filename="example_inspiral", folder=EXAMPLE_DATA)

INPUT_LABELS = ["Spin", "Mass Ratio", "Endflag", "Target Error", 
                "Label", "Constants", "Orientation Velocity",
                "Tetrad Velocity", "Kepler Parameters", "Position",
                "True Velocity"]
for label, entry in zip(INPUT_LABELS, loaded_orbit["inputs"]):
    print(f"{label}: {entry}")
print()
     
fig1, ax1 = plotvalues([loaded_orbit], "energy")
fig2, ax2 = plotvalues([loaded_orbit], "ecc", "semilat")
plt.show()

# The output of EMRIGenerator is pretty big, especially for longer datasets.
# To keep from filling up our memory too quickly, we only save a portion of the data
#   and reconstruct the rest when loading it.
# If those derived quantities aren't needed, reconstruct=False skips that 
#   additional processing and memory allocation.
short_load = ml.load_emri_data(filename="example_inspiral", folder=EXAMPLE_DATA, reconstruct=False)

print("Essential data:", list(short_load.keys()))
print("Reconstructed data:", sorted(set(loaded_orbit.keys()) - set(short_load.keys())))

# If index.json is missing or outdated (for example, after manually
# copying HDF5 files into a directory), it can be rebuilt:
#
# index = ml.rebuild_index(folder=EXAMPLE_DATA, overwrite=True)