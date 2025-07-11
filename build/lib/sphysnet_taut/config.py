import os
from importlib.resources import files

rootp = os.path.abspath(os.path.dirname(__file__))
smarts_path = os.path.join(rootp, "smarts_vmr.txt")

transform_path = files('sphysnet_taut.data').joinpath("smirks_transform_all.txt")
# transform_path = os.path.join(rootp, "smirks_tansform_all.txt")


checkpoints = ["best_model1.pt",
               "best_model2.pt",
               "best_model3.pt",
               "best_model4.pt",
               "best_model5.pt"]

# model_paths = [os.path.join(rootp, "weights", path) for path in model_paths]
model_paths = [files('sphysnet_taut.weights').joinpath(_) for _ in checkpoints]
