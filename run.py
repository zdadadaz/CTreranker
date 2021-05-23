from utils import tool
from eval import trec_eval

# tool.check_xml_tag_parallel('../../data/TRECPM2019')
# tool.combine_all_tags('upload/', '2019_all')

# tool.check_xml_tag_parallel('../../data/TRECCT2021')
# tool.combine_all_tags('upload/', '2021_all')

# tool.check_xml_tag_parallel('../../data/test_collection/clinicaltrials_xml')
# tool.combine_all_tags('upload/', '2016_tc_all')


dirs = ['cv{}'.format(str(i)) for i in range(2)]
trec_eval.average_all_cv_eval('output_ts/bs_noName_pretrained_base', dirs, 'overall_cv.eval')