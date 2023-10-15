import os
import json
import shutil
import argparse
import pycomet

PATH_TO_CONFIG = os.path.dirname(pycomet.__file__) + "/config/"


def create_config():
  """
  Creates a temporary configuration file.

  This function creates a temporary copy of the configuration
  file, allowing for temporary modifications or experimentation
  without affecting the original configuration.
  """
  shutil.copyfile(
    PATH_TO_CONFIG + "/configfile.json",
    PATH_TO_CONFIG + "/configfile_tmp.json"
  )


def read_config():
  """
  Reads the configuration file and returns it as an argparse.Namespace object.

  This function reads the JSON configuration file and converts its
  contents into an argparse.Namespace object. It is designed for
  handling configuration settings in your application.

  :return: An argparse.Namespace object containing the configuration data.
  :rtype: argparse.Namespace
  """
  with open(PATH_TO_CONFIG + "/configfile_tmp.json", 'r') as file:
    cfg = argparse.Namespace(**json.load(file))
  return cfg


def update_config(
  key,
  value
):
  """
  Update a temporary configuration file with a new key-value pair.

  :param key: The configuration key to be updated or added.
  :type key: str

  :param value: The value associated with the specified key.
  :type value: any

  This function reads an existing temporary configuration file, updates or
  adds a new key-value pair, and saves the changes back to the same file.
  If the specified key already exists, its value is replaced with the new
  value. If the key is not present, a new key-value pair is added.
  """
  with open(PATH_TO_CONFIG + "/configfile_tmp.json", "r+") as file:
    data = json.load(file)
    data[key] = value
    file.seek(0)
    json.dump(data, file, indent=2)
    file.truncate()
