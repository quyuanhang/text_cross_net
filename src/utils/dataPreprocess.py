import sys
import sklearn
import pandas as pd
import argparse
import collections
from tqdm import tqdm
import re
import numpy as np

expect_features = [
    'expect_id',
    'geek_id',
    'l3_name',
    'city',
    'gender',
    'degree',
    'fresh_graduate',
    'apply_status',
    'completion',
    'cv'
]

job_features = [
    'job_id',
    'boss_id',
    'position',
    'city',
    'degree',
    'experience',
    'area_business_name',
    'boss_title',
    'is_hr',
    'stage',
    'jd'
]

def dataFormat(fp, feature_name, word_dict, one_hot=True):
    n_feature = len(feature_name)
    with open(fp) as f:
        data = f.read().strip().split('\n')
    print('split raw data ...')
    ids_list, category_features_list, docs_list = [], [], []
    for line in tqdm(data):
        features = line.split('\001')
        if len(features) != n_feature:
            continue
        # id
        cid = features[0]
        ids_list.append(cid)
        # category feature
        category_features = features[1:-1]
        if one_hot:
            category_features = dict(zip(feature_name[1:-1], category_features))
        category_features_list.append(category_features)
        # text feature
        doc = re.split('[\001\n\t ]+', features[-1].strip())[:100]
        doc = [word_dict.get(word, 0) for word in doc]
        docs_list.append(doc)
    id_to_row = {k: v for v, k in enumerate(ids_list)}
    docs_matrix = np.array(docs_list)
    if one_hot:
        vec = sklearn.feature_extraction.DictVectorizer()
        category_features_matrix = vec.fit_transform(category_features_list)
        category_features_name = vec.get_feature_names()
        return id_to_row, category_features_matrix, category_features_name, docs_matrix, word_dict
    else:
        return id_to_row, category_features_list, docs_matrix, word_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datain', default='../Data/multi_data/multi_data')
    parser.add_argument('--dataout', default='./data/interview')
    args = parser.parse_args()




