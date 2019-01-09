
from . import ansi
from .progress import Progress
import os

def find_files(paths, exts, recurse=True, quiet=False):
    """search paths for files ending with exts"""
    if recurse is True:
        recurse = float('inf')
    
    file_paths = []
    def search_dir(path,r):
        for f in os.listdir(path):
            f = os.path.join(path, f)
            if os.path.isdir(f) and r < recurse:
                search_dir(f, r+1)
            elif f.endswith(exts):
                file_paths.append(f)
    
    with Progress('Building file list',len(paths),quiet) as p:
        for p in paths:
            if os.path.isdir(p):
                with Progress('Searching {} for {} files'.format(p, exts),quiet=quiet):
                    search_dir(p,0)
            elif p.endswith(exts):
                file_paths.append(p)
    return file_paths

