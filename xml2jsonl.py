import os
import xml.etree.ElementTree as ET
import pathlib
import json 
import re

root = '../../data/TRECPM2017/clinicaltrials_xml/'
out_path = '../../data/TRECPM2017/clinicaltrials_json/'
# out_path = root.replace('clinicaltrials_xml','clinicaltrials_json')
year = '2017'

filelist = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files]
for f in filelist:
    fn = f.split('/')[-1][:-4]
    outname = f.replace('clinicaltrials_xml', 'clinicaltrials_json')
    print(f,fn)
    if fn[0] == '.':
        continue
    # pathlib.Path(outname).mkdir(parents=True, exist_ok=True)
    tree = ET.parse(f)
    root = tree.getroot()
    res = {}
    res['year'] = year
    res['id'] = fn
    for child in root:
        if child.tag.lower()=='brief_title':
            res['bt'] = re.sub(r"[\n\t\r]*", "",root[2].text)
            res['bt'] = re.sub(r"  ", "",res['bt'])
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
                    fulltext = c[0].text.lower().split('criteria')
                    if len(fulltext)>=2:
                        res['inclusion'] = re.sub(r"[\n\t\r]*", "", ''.join(fulltext[1].split('\n\n')[1:-1])) #''.join(fulltext[1].split('\n\n')[1:-1])
                        res['inclusion'] = re.sub(r"  ", "",res['inclusion'])
                    if len(fulltext)>=3:
                        res['exclusion'] = re.sub(r"[\n\t\r]*", "", ''.join(fulltext[2].split('\n\n')[1:]))#''.join(fulltext[2].split('\n\n')[1:])
                        res['exclusion'] = re.sub(r"  ", "",res['exclusion'])
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

    