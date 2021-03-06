import os
import sys
import importlib
import json
import numpy as np
from functools import lru_cache

############################################
###########################################
#
#
# 	 FILE SYS HELPERS
#
#
###########################################
###########################################


def getFileDir():
	return os.path.dirname(os.path.abspath(__file__))

# def #CREATE RELATIVE IMPORT FUNCTION TODO

# def get_relative_dir(rel_path):
#     FILE_DIR = os.path.dirname('__file__')
#     rel = os.path.abspath(os.path.join(FILE_DIR, '../'))
#     return rel
# def relative_import(mod_name, rel_path):
#     rel = get_relative_dir(rel_path)
#     sys.path.append(rel)
#     return importlib.import_module(mod_name)

def get_relative_dir(rel_path):
    try:
        FILE_DIR = os.path.dirname('__file__')
    except:
        FILE_DIR = os.path.dirname(__file__)
    if FILE_DIR == '':
        FILE_DIR = os.path.dirname(__file__)
    rel = os.path.abspath(os.path.join(FILE_DIR, rel_path))
    return rel

def relative_import(mod_name, rel_path):
    rel = get_relative_dir(rel_path)
    if rel not in sys.path:
        sys.path.append(rel)
    return importlib.import_module(mod_name)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj): #pylint: disable=method-hidden
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)    


@lru_cache(maxsize=1)
def git_root():
    from subprocess import check_output, CalledProcessError
    ''' returns the absolute path of the repository root '''
    try:
        base = check_output('git rev-parse --show-toplevel', shell=True)
    except CalledProcessError:
        raise IOError('Current working directory is not a git repository')
    return base.decode('utf-8').strip()
# ROOT_DIR = git_root()        