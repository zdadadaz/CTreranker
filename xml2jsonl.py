import os
import xml.etree.ElementTree as ET
import pathlib
import json 
import re
import argparse
from multiprocessing import Pool, cpu_count

year = '2019'
def save_json(f):
    fn = f.split('/')[-1][:-4]
    out_path = '/'.join(f.replace('clinicaltrials_xml', 'clinicaltrials_json_bt').split('/')[:-1])
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    # print('test',out_path, fn)
    # print(f,fn)
    tree = ET.parse(f)
    root = tree.getroot()
    res = {'gender':'', 'min_age':'', 'max_age':'','criteria':''}
    res['year'] = year
    res['id'] = fn
    for child in root:
        if child.tag.lower()=='brief_title':
            res['contents'] = re.sub(r"[\n\t\r]*", "",root[2].text)
            res['contents'] = re.sub(r"  ", "",res['contents'])
        elif child.tag.lower()=='brief_summary':
            res['bs'] = re.sub(r"[\n\t\r]*", "",child[0].text)
            res['bs'] = re.sub(r"  ", "",res['bs'])
        elif child.tag.lower()=='detailed_description':
            res['dd'] = re.sub(r"[\n\t\r]*", "",child[0].text)
            res['dd'] = re.sub(r"  ", "",res['dd'])
        elif child.tag.lower()=='primary_outcome':
            tmp = []
            for c in child:
                tmp.append(c.text)
            if 'primary_outcome' in res:
                res['primary_outcome'] = res['primary_outcome'] + ', '.join(tmp)
            else:
                res['primary_outcome'] = ', '.join(tmp)
        elif child.tag.lower()=='intervention':
            res['intervention_type'] = child[0].text
            res['intervention_name'] = child[1].text
        elif child.tag.lower()=='eligibility':
            for c in child:
                if c.tag.lower() == 'criteria':
                    res['criteria'] = re.sub(r"[\n\t\r]*", "", c[0].text.lower().strip())
                    res['criteria'] = re.sub(r"  ", "", res['criteria'])
                elif c.tag.lower() =='gender':
                    res['gender'] = c.text
                elif c.tag.lower()=='minimum_age':
                    res['min_age'] = c.text[:-6] if c.text!='N/A' else 'N/A'
                elif c.tag.lower()=='maximum_age':
                    res['max_age'] = c.text[:-6] if c.text!='N/A' else 'N/A'
                elif c.tag.lower()=='healthy_volunteers':
                    res['healthy'] = c.text
    json_object = json.dumps(res,indent=2)
    with open(os.path.join(out_path,fn+'.json'), "w") as outfile:
        outfile.write(json_object)

def main():
    args= argretrieve()
    root = args.IPath
    out_path = args.OPath
    global year
    year = args.yr
    # root = '../../data/TRECPM2019/clinicaltrials_xml/'
    # out_path = '../../data/TRECPM2019/clinicaltrials_json_bt/'

    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    filelist = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.split('.')[-1] == 'xml':
                filelist.append(os.path.join(path, name))
    pool = Pool(processes=(cpu_count() - 1))
    pool.map(save_json, filelist)
    pool.close()

def argretrieve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--IPath', help="Source Path for indexing")
    parser.add_argument('--OPath', help="Out Path for indexing")
    parser.add_argument('--yr', help="year of TREC PM")
    return parser.parse_args()

if __name__ == '__main__':
    main()