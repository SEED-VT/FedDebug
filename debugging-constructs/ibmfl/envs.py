"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
import os
import pathlib
from environs import Env

env = Env()
env.read_env()

# export FL_WORKING_DIR=/home/username/
# export FL_MODEL_DIR=/home/username/model


def create_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


with env.prefixed("FL_"):
    working_directory = env("WORKING_DIR", os.getcwd())
    create_dir(working_directory)

    model_directory = env("MODEL_DIR", working_directory)
    create_dir(model_directory)
