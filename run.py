import pandas as pd

from utils import tool
from eval import trec_eval

# tool.check_xml_tag_parallel('../../data/TRECPM2019')
# tool.combine_all_tags('upload/', '2019_all')

# tool.check_xml_tag_parallel('../../data/TRECCT2021')
# tool.combine_all_tags('upload/', '2021_all')

# tool.check_xml_tag_parallel('../../data/test_collection/clinicaltrials_xml')
# tool.combine_all_tags('upload/', '2016_tc_all')


# dirs = ['cv{}'.format(str(i)) for i in range(2)]
# trec_eval.average_all_cv_eval('output_ts/bs_noName_pretrained_base', dirs, 'overall_cv.eval')

# import os
# import pandas
# filelist = []
# root = 'output_ts'
# df_out = pd.DataFrame(columns=['method', 'map', 'Rprec', 'recip_rank', 'P_5', 'P_10', 'P_15', 'ndcg'])
# for path, subdirs, files in os.walk(root):
#     for name in files:
#         if name[0] != '.' and name == 'overall_cv.eval.csv':
#             # print(os.path.join(path, name))
#             df_out = df_out.append(pd.read_csv(os.path.join(path, name)))
#             # filelist.append(os.path.join(path, name))
# df_out.to_csv(root + '/overall_disease_ts.csv', index=False)

from utils import simpleretrieve

if __name__ == '__main__':
    simpleretrieve.simple()

