import os
import xml.etree.ElementTree as ET
import pathlib
import json
import re
import argparse
from multiprocessing import Pool, cpu_count

year = '2017'
outfilename = None


def save_json(f):
    fn = f.split('/')[-1][:-4]
    out_path = '/'.join(f.replace('clinicaltrials_xml', outfilename).split('/')[:-1])
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    # print('test',out_path, fn)
    # print(f,fn)
    tree = ET.parse(f)
    root = tree.getroot()
    res = {'gender': '', 'min_age': '', 'max_age': '', 'criteria': ''}
    res['year'] = year
    res['id'] = fn
    res['contents'] = ''
    for child in root:
        if child.tag.lower() == 'brief_title':
            res['bt'] = re.sub(r"[\n\t\r]*", "", root[2].text)
            res['bt'] = re.sub(r"  ", "", res['bt'])
            res['contents'] += ' ' + res['bt']
        elif child.tag.lower() == 'brief_summary':
            res['bs'] = re.sub(r"[\n\t\r]*", "", child[0].text)
            res['bs'] = re.sub(r"  ", "", res['bs'])
            res['contents'] += ' ' + res['bs']
        elif child.tag.lower() == 'detailed_description':
            res['dd'] = re.sub(r"[\n\t\r]*", "", child[0].text)
            res['dd'] = re.sub(r"  ", "", res['dd'])
            res['contents'] += ' ' + res['dd']
        elif child.tag.lower() == 'official_title':
            if len(child.text)!= 0:
                res['ot'] = re.sub(r"[\n\t\r]*", "", child.text)
                res['ot'] = re.sub(r"  ", "", res['ot'])
                res['contents'] += ' ' + res['ot']
        elif child.tag.lower() == 'keyword':
            if len(child.text) != 0:
                tmp = re.sub(r"[\n\t\r]*", "", child.text)
                tmp = re.sub(r"  ", "", tmp)
                if 'kw' not in res:
                    res['kw'] = tmp
                else:
                    res['kw'] += ' ' + tmp
                res['contents'] += ', ' + tmp
        elif child.tag.lower() == 'primary_outcome':
            tmp = []
            for c in child:
                tmp.append(c.text)
            if 'primary_outcome' in res:
                res['primary_outcome'] = res['primary_outcome'] + ', '.join(tmp)
            else:
                res['primary_outcome'] = ', '.join(tmp)
            res['contents'] += ' ' + res['primary_outcome']
        elif child.tag.lower() == 'intervention':
            if 'intervention_type' in res:
                res['intervention_type'] += ' ' + child[0].text
                res['intervention_name'] += ' ' + child[1].text
            else:
                res['intervention_type'] = child[0].text
                res['intervention_name'] = child[1].text
            res['contents'] += ' ' + child[0].text
        elif child.tag.lower() == 'eligibility':
            for c in child:
                if c.tag.lower() == 'criteria':
                    # res['criteria'] = re.sub(r"[\n\t\r]*", "", c[0].text.lower().strip())
                    # res['criteria'] = re.sub(r"  ", "", res['criteria'])
                    # res['contents'] += ' ' + res['criteria']
                    res['criteria'] = c[0].text
                elif c.tag.lower() == 'gender':
                    res['gender'] = c.text
                    res['contents'] += ' ' + res['gender']
                elif c.tag.lower() == 'minimum_age':
                    if 'year' in c.text.lower():
                        res['min_age'] = c.text.split(' ')[0]
                    else:
                        res['min_age'] = c.text[:-6] if c.text != 'N/A' else 'N/A'
                    # need remove after test
                    # res['min_age'] = c.text
                    res['contents'] += ' ' + c.text
                elif c.tag.lower() == 'maximum_age':
                    if 'year' in c.text.lower():
                        res['max_age'] = c.text.split(' ')[0]
                    else:
                        res['max_age'] = c.text[:-6] if c.text != 'N/A' else 'N/A'
                    # need remove after test
                    # res['max_age'] = c.text
                    res['contents'] += ' ' + c.text
        elif child.tag.lower() == 'condition':
            if 'condition' not in res:
                res['condition'] = child.text
            else:
                res['condition'] += ', ' + child.text
            res['contents'] += ' ' + child.text

    json_object = json.dumps(res, indent=2)
    with open(os.path.join(out_path, fn + '.json'), "w") as outfile:
        outfile.write(json_object)

def main():
    args = argretrieve()
    root = args.IPath
    out_path = args.OPath

    # root = '../../data/test_collection/clinicaltrials_xml_new/'
    # out_path = '../../data/test_collection/clinicaltrials_json/'

    global year
    global outfilename
    year = args.yr
    if len(out_path.split('/')[-1]) > 0:
        outfilename = out_path.split('/')[-1]
    else:
        outfilename = out_path.split('/')[-2]

    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    filelist = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[0] != '.' and name.split('.')[-1] == 'xml': # and name == 'NCT00046436.xml':
                filelist.append(os.path.join(path, name))

    pool = Pool(processes=(cpu_count() - 1))
    pool.map(save_json, filelist)
    pool.close()



def argretrieve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--IPath', help="Source Path for indexing", default=None)
    parser.add_argument('--OPath', help="Out Path for indexing", default=None)
    parser.add_argument('--yr', help="year of TREC PM", default='2016')
    return parser.parse_args()


if __name__ == '__main__':
    main()
