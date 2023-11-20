from utils import load_model_by_name
from dadvi.pymc.utils import get_unconstrained_variable_names
import json


text_file = "./all_arm_names.txt"

lookup = dict()

for line in open(text_file):

    print(line.strip())

    cur_model = load_model_by_name(line.strip())
    names = get_unconstrained_variable_names(cur_model)
    lookup[line.strip()] = names

for other_model in ["microcredit", "potus", "occ_det", "tennis"]:

    print(other_model)

    cur_model = load_model_by_name(other_model)
    names = get_unconstrained_variable_names(cur_model)
    lookup[other_model] = names

json.dump(lookup, open("./unconstrained_lookup.json", "w"))
