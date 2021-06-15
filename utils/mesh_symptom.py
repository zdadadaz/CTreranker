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
# import pymedtermino
# pymedtermino.LANGUAGE = "en"
# pymedtermino.REMOVE_SUPPRESSED_CONCEPTS = True
# from pymedtermino.all import *
# from pymedtermino import *
# from pymedtermino.snomedct import *


with open('mesh/mesh_name2tree.json') as f:
    mesh_name2tree = json.load(f)

with open('mesh/mesh_tree2synonym.json') as f:
    mesh_tree2synonym = json.load(f)


def mesh_fn(f):
    fn = f.split('/')[-1][:-4]
    out_path = '/'.join(f.replace('clinicaltrials_xml_new', outfilename).split('/')[:-1])
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
            if 'intervention_type' in res:
                res['intervention_type'] += ', ' + child[0].text
                res['intervention_name'] += ', ' + child[1].text
            else:
                res['intervention_type'] = child[0].text
                res['intervention_name'] = child[1].text
            res['contents'] += ', ' + child[0].text + ', ' + child[1].text
        elif child.tag.lower() == 'eligibility':
            for c in child:
                if c.tag.lower() == 'criteria':
                    if 'criteria:' in c[0].text.lower():
                        try:
                            tmp = re.sub(r"[\n\t\r]*", "", c[0].text.lower().strip()).split('criteria:')
                            if 'inclusion' in tmp[0].lower():
                                arr_tmp = []
                                for inc in tmp[1].split('    '):
                                    if len(inc.strip()) != 0:
                                        if 'exclusion' in inc:
                                            continue
                                        else:
                                            arr_tmp.append(inc.strip().strip('-').strip())
                                res['inclusion'] = '# '.join(arr_tmp)
                            if 'exclusion' in tmp[1].lower() and len(tmp) >= 3:
                                arr_tmp = []
                                for exc in tmp[2].split('    '):
                                    if len(exc.strip()) != 0:
                                        arr_tmp.append(exc.strip().strip('-').strip())
                                res['exclusion'] = '# '.join(arr_tmp)
                        except:
                            print(fn, 'criteria error')
                    res['criteria'] = re.sub(r"[\n\t\r]*", "", c[0].text.lower().strip())
                    res['criteria'] = re.sub(r"  ", "", res['criteria'])
                    res['criteria'] = re.sub(r"exclusion", " exclusion", res['criteria'])
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
        elif child.tag.lower() == 'condition':
            condition = child.text
            if 'condition' not in res:
                res['condition'] = condition
            else:
                res['condition'] += ' | ' + condition
            res['contents'] += ' ' + condition
            summary, symptoms = [], []

            try:
                # wiki info
                search_res = wikipedia.search(condition, results=2)
                so = wptools.page(search_res[0], silent=True).get_parse()
                # print(wikipedia.summary(search_res[0], sentences=10))
                summary.append(wikipedia.summary(search_res[0], sentences=10))
                if 'symptoms' in so.data['infobox']:
                    # print(so.data['infobox']['symptoms'].strip().replace('[', '').replace(']', ''))
                    symptoms.append(so.data['infobox']['symptoms'].strip().replace('[', '').replace(']', ''))
            except:
                pass
            if len(symptoms) != 0:
                if 'wiki_symptoms' not in res:
                    res['wiki_symptoms'] = ', '.join(symptoms)
                else:
                    res['wiki_symptoms'] += ' | ' + ', '.join(symptoms)
            if len(summary) != 0:
                if 'wiki_summary' not in res:
                    res['wiki_summary'] = ', '.join(summary)
                else:
                    res['wiki_summary'] += ' | ' + ', '.join(summary)

        # elif child.tag.lower() == 'condition_browse':
        #     symptoms = []
        #     summary = []
        #     caption = []
        #     synonyms = set()
        #     res['mesh'] = ''
        #     for c in child:
        #         res['mesh'] += ', ' + c.text
        #         try:
        #             # mesh synonym
        #             if c.text.lower() in mesh_name2tree:
        #                 if ',' in mesh_name2tree[c.text.lower()]:
        #                     for tree in mesh_name2tree[c.text.lower()].split(','):
        #                         for syn in mesh_tree2synonym[tree]:
        #                             synonyms.add(syn)
        #                 else:
        #                     tree = mesh_name2tree[c.text.lower()]
        #                     for syn in mesh_tree2synonym[tree]:
        #                         synonyms.add(syn)
        #             # wiki info
        #             search_res = wikipedia.search(c.text, results=2)
        #             so = wptools.page(search_res[0], silent=True).get_parse()
        #             summary.append(wikipedia.summary(search_res[0], sentences=3))
        #             if 'caption' in so.data['infobox']:
        #                 caption.append(so.data['infobox']['caption'].strip().replace('[', '').replace(']', ''))
        #             if 'symptoms' in so.data['infobox']:
        #                 symptoms.append(so.data['infobox']['symptoms'].strip().replace('[', '').replace(']', ''))
        #         except:
        #             pass
        #             # print(c.text, wikipedia.search(c.text, results=3))
        #     if len(symptoms) != 0:
        #         res['wiki_symptoms'] = ', '.join(symptoms)
        #     if len(summary) != 0:
        #         res['wiki_summary'] = ', '.join(summary)
        #     if len(caption) != 0:
        #         res['wiki_caption'] = ', '.join(caption)
        #     if len(synonyms) != 0:
        #         res['synonyms'] = ', '.join(list(synonyms))
    # if 'inclusion' in res:
    #     print('inclusion', res['inclusion'])
    #     print('\n')
    # if 'exclusion' in res:
    #     print('exclusion', res['exclusion'])
    #     print('\n')
    # if 'criteria' in res:
    #     print('criteria', res['criteria'])
    #     print('\n')

    json_object = json.dumps(res, indent=2)
    with open(os.path.join(out_path, fn + '.json'), "w") as outfile:
        outfile.write(json_object)


def main():
    global outfilename
    outfilename = 'clinicaltrials_json_cond_sym_new'
    out_path = f'../../data/test_collection/{outfilename}/'
    root = '../../data/test_collection/clinicaltrials_xml_new/'
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    filelist = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[0] != '.' and name.split('.')[-1] == 'xml':  # and name == 'NCT00046436.xml':
                filelist.append(os.path.join(path, name))

    # mesh_fn(filelist[2222])
    pool = Pool(processes=(cpu_count() - 1))
    pool.map(mesh_fn, filelist)
    pool.close()


if __name__ == '__main__':
    main()
