from utils.demographic import get_topic_demographic
from utils import readfile as rf

def demo_test():
    root_path = '../../data/test_collection'
    query_types = ['bs'] #, 'dd']
    for query_type in query_types:
        query_dict = {}
        if 'bs' in query_type:
            rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-summary.topics')
        if 'dd' in query_type:
            rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-description.topics')
        for i in query_dict:
            gender, age = get_topic_demographic(query_dict[i]['text'])
            print(i, query_type, gender, age)