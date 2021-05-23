import os
import xml.etree.ElementTree as ET
import pathlib
import json
import re
import argparse
from multiprocessing import Pool, cpu_count
import random


def check_xml_tag_list(filelist):
    output_path = 'upload/' + str(random.randint(0,1000000000))
    # print(output_path)
    res = set()
    for f in filelist:
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            res.add(child.tag.lower())
    out = []
    for i in res:
        out.append(i + '\n')
    with open(output_path + '.tags', "w") as f:
        f.writelines(out)

def read_tag_file(path_to_tag):
    res = set()
    with open(path_to_tag, 'r') as f:
        contents = f.readlines()

    for line in contents:
        tag = line.strip()
        res.add(tag)
    return res

def combine_all_tags(path_to_folder, out_path):
    filelist = []
    for path, subdirs, files in os.walk(path_to_folder):
        for name in files:
            if name.split('.')[-1] == 'tags':
                filelist.append(os.path.join(path, name))
    res = None
    for f in filelist:
        if not res:
            res = read_tag_file(f)
        else:
            res.union(read_tag_file(f))

    out = []
    for i in res:
        out.append(i + '\n')
    with open(out_path + '.tags', "w") as f:
        f.writelines(out)


def check_xml_tag_parallel(root):
    filelist = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.split('.')[-1] == 'xml' and name[0] != '.':
                filelist.append(os.path.join(path, name))
    split = cpu_count() - 1
    arr = []
    step = len(filelist) // split
    for i in range(split):
        if i == split - 1:
            arr.append(filelist[(step * i):])
        else:
            arr.append(filelist[(step * i):(step * (i + 1))])

    pool = Pool(processes=split)
    pool.map(check_xml_tag_list, arr)
    pool.close()
