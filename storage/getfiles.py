import os 
from os.path import isfile, join

def get_files(path):
    files = []
    for f in os.listdir(path):
        if isfile(join(path, f)):
            files.append(join(path, f))
        else:
            [files.append(k) for k in get_files(join(path, f))]
    return files

def get_file_sizes(files): return [os.path.getsize(f) for f in files]

def get_fpf_and_opf_size(path):
    files = get_files(path)
    fpf = [f for f in files if f[-8:] == "4p4km.nc"]
    opf = [f for f in files if f[-8:] == "1p5km.nc"]
    return sum(get_file_sizes(fpf)), sum(get_file_sizes(opf))

def get_average_file_size(path):
    files = get_files(path)
    fpf_size = get_file_sizes([f for f in files if f[-8:] == "4p4km.nc"])
    opf_size = get_file_sizes([f for f in files if f[-8:] == "1p5km.nc"])
    return sum(fpf_size)/len(fpf_size), sum(opf_size)/len(opf_size)

PATH = "../../../work/br19097"
fpf, opf = get_fpf_and_opf_size(PATH)

print(f"4.4 km data size: {fpf / 1000**3} Gb, 1.5 data size: {opf / 1000**3}")

fpf_mean, opf_mean = get_average_file_size(PATH)

print(f"4.4 km average size: {fpf_mean / 1000**3} Gb, 1.5 average size: {opf_mean / 1000**3}")