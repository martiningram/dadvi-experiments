from glob import glob
from tqdm import tqdm
from os.path import split
import subprocess
import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--ARM-folder", required=True, type=str)
parser.add_argument("--target-dir", required=False, type=str, default="./")
parser.add_argument(
    "--convert-script-path", required=False, type=str, default="./convert.R"
)
parser.add_argument(
    "--allow-overwrite", required=False, default=False, action="store_true"
)
args = parser.parse_args()

data_files = glob(os.path.join(args.ARM_folder, "*/*.data.R"))
target_dir = args.target_dir
convert_script_path = args.convert_script_path

assert len(data_files) > 0, "Unable to find any data files in the ARM folder!"
assert os.path.isfile(
    convert_script_path
), f"Conversion script not found at {convert_script_path}!"

failures = open(os.path.join(target_dir, "conversion_failures.txt"), "w")

for cur_data_file in tqdm(data_files):

    cur_chapter_name = cur_data_file.split("/")[-2]
    cur_target_dir = os.path.join(target_dir, cur_chapter_name)
    os.makedirs(cur_target_dir, exist_ok=True)

    cur_filename = split(cur_data_file)[-1]
    target_name = cur_filename.rstrip(".R")
    target_name = target_name + ".json"
    target_path = os.path.join(cur_target_dir, target_name)

    command_list = [
        "Rscript",
        convert_script_path,
        "--input-file",
        cur_data_file,
        "--output-file",
        target_path,
    ]

    if args.allow_overwrite:
        command_list = command_list + ["--allow-overwrite"]

    try:
        result = subprocess.check_output(command_list)
    except Exception as e:
        # TODO Maybe print exception to file too to investigate
        print(f"Failed to convert {cur_data_file} with exception: {e}")
        print(cur_data_file, file=failures, flush=True)
        continue

failures.close()
