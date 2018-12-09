from sklearn import feature_extraction
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
    ids_list, category_features_list, docs_list, raw_doc_list = [], [], [], []
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
        # doc = re.split('[\001\n\t ]+', features[-1].strip())[:100]
        doc = re.split('[\001\n\t ]+', features[-1].strip())
        raw_doc_list.append(doc)
        doc = [word_dict.get(word, 0) for word in doc]
        docs_list.append(doc)
    id_to_row = {k: v for v, k in enumerate(ids_list)}
    docs_matrix = np.array(docs_list)
    if one_hot:
        vec = feature_extraction.DictVectorizer()
        category_features_matrix = vec.fit_transform(category_features_list).todok()
        category_features_name = vec.get_feature_names()
        return id_to_row, category_features_matrix, category_features_name, docs_matrix, raw_doc_list
    else:
        return id_to_row, category_features_list, docs_matrix, raw_doc_list

def build_dict(fps, w_freq):
    words = []
    for fp in fps:
        with open(fp) as f:
            for line in tqdm(f):
                line = line.strip().split('\001')[-1]
                line = line.split()
                words.extend(line)
    words_freq = collections.Counter(words)
    word_dict = {k: v for k, v in words_freq.items() if v >= w_freq}
    word_dict = {k: v for v, k in enumerate(word_dict.keys(), 2)}
    word_dict['__pad__'] = 0
    word_dict['__unk__'] = 1
    print('n_words: {}'.format(len(word_dict)))
    return word_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datain', default='../Data/multi_data/multi_data')
    parser.add_argument('--dataout', default='./data/interview')
    parser.add_argument('--word_freq', type=int, default=5)
    args = parser.parse_args()

    fps = ['{}.profile.job'.format(args.datain), '{}.profile.expect'.format(args.datain)]
    word_dict = build_dict(fps, args.word_freq)
    idx, cfs, cfn, docs = dataFormat(
        fp='{}.profile.job'.format(args.datain),
        feature_name=job_features,
        word_dict=word_dict,
        one_hot=True
    )

    print('done')



