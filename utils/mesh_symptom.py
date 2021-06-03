import os
# os.system('pip install wikipedia wptools')
import xml.etree.ElementTree as ET
import pathlib
import json
import re
from multiprocessing import Pool, cpu_count
import wikipedia
import wptools
wikipedia.set_lang("En")
outfilename = None

def mesh_fn(f):
    fn = f.split('/')[-1][:-4]
    out_path = '/'.join(f.replace('clinicaltrials_xml', outfilename).split('/')[:-1])
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    tree = ET.parse(f)
    root = tree.getroot()
    res = {'gender': '', 'min_age': '', 'max_age': '', 'criteria': ''}
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
            if len(child.text) != 0:
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
            res['intervention_type'] = child[0].text
            res['intervention_name'] = child[1].text
            res['contents'] += ' ' + res['intervention_type']
        elif child.tag.lower() == 'eligibility':
            for c in child:
                if c.tag.lower() == 'criteria':
                    res['criteria'] = re.sub(r"[\n\t\r]*", "", c[0].text.lower().strip())
                    res['criteria'] = re.sub(r"  ", "", res['criteria'])
                    res['contents'] += ' ' + res['criteria']
                elif c.tag.lower() == 'gender':
                    res['gender'] = c.text
                    res['contents'] += ' ' + res['gender']
                elif c.tag.lower() == 'minimum_age':
                    if 'year' in c.text.lower():
                        res['min_age'] = c.text.split(' ')[0]
                    else:
                        res['min_age'] = c.text[:-6] if c.text != 'N/A' else 'N/A'
                    res['contents'] += ' ' + c.text
                elif c.tag.lower() == 'maximum_age':
                    if 'year' in c.text.lower():
                        res['max_age'] = c.text.split(' ')[0]
                    else:
                        res['max_age'] = c.text[:-6] if c.text != 'N/A' else 'N/A'
                    res['contents'] += ' ' + c.text
        elif child.tag.lower() == 'condition_browse':
            symptoms = []
            for c in child:
                try:
                    search_res = wikipedia.search(c.text, results=3)
                    so = wptools.page(search_res[0], silent=True).get_parse()
                    infobox = so.data['infobox']['symptoms']
                    for j in infobox.split(','):
                        symptoms.append(j.strip().replace('[','').replace(']',''))
                except:
                    pass
                    # print(c.text, wikipedia.search(c.text, results=3))
            if len(symptoms) != 0:
                res['contents'] += ' ' + ', '.join(symptoms)
                res['symptoms'] = ', '.join(symptoms)
    json_object = json.dumps(res, indent=2)
    with open(os.path.join(out_path, fn + '.json'), "w") as outfile:
        outfile.write(json_object)


def main():
    global outfilename
    outfilename = 'clinicaltrials_json_sym'
    root = '../../data/test_collection/clinicaltrials_xml/'
    out_path = root.replace('clinicaltrials_xml', outfilename)
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    filelist = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[0] != '.' and name.split('.')[-1] == 'xml': # and name == 'NCT00046436.xml':
                filelist.append(os.path.join(path, name))
    # mesh_fn(filelist[1])
    pool = Pool(processes=(cpu_count() - 1))
    pool.map(mesh_fn, filelist)
    pool.close()


if __name__ == '__main__':
    main()