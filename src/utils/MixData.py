import sys
import re
import collections
import numpy as np
from tqdm import tqdm
from sklearn import feature_extraction
from scipy import sparse


class MixData:
    def __init__(self, fpin, fpout, wfreq, doc_len):
        self.fpin = fpin
        self.fpout = fpout
        self.doc_len = doc_len
        exp_features_names = [
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
        job_features_names = [
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
        fps = [
            '{}.profile.job'.format(fpin),
            '{}.profile.expect'.format(fpin)]
        self.word_dict, self.word_list = self.build_dict(fps, wfreq)

        require_exp_feature = exp_features_names[2:-1]
        require_job_feature = job_features_names[2:-1]
        self.feature_name = require_exp_feature + require_job_feature
        self.exp_to_row, self.exp_feature_dicts, self.exp_features, \
            exp_features_names_sparse, self.exp_docs, self.exp_doc_raw = \
            self.build_features(
                fp='{}.profile.expect'.format(fpin),
                feature_name=exp_features_names,
                requir_feature_name=require_exp_feature,
            )
        self.job_to_row, self.job_feature_dicts, self.job_features, \
            job_features_names_sparse, self.job_docs, self.job_doc_raw = \
            self.build_features(
                fp='{}.profile.job'.format(fpin),
                feature_name=job_features_names,
                requir_feature_name=require_job_feature,
            )
        self.feature_name_sparse = job_features_names_sparse + exp_features_names_sparse

    @staticmethod
    def build_dict(fps, w_freq):
        words = []
        for fp in fps:
            with open(fp) as f:
                for line in tqdm(f):
                    line = line.strip().split('\001')[-1]
                    line = line.split()
                    words.extend(line)
        words_freq = collections.Counter(words)
        word_list = [k for k, v in words_freq.items() if v >= w_freq]
        word_list = ['__pad__', '__unk__'] + word_list
        word_dict = {k: v for v, k in enumerate(word_list)}
        print('n_words: {}'.format(len(word_dict)), len(word_list))
        return word_dict, word_list

    def build_features(self, fp, feature_name, requir_feature_name, one_hot=True):
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
            if one_hot:
                features_dict = dict(zip(feature_name, features))
                features_dict = {k: features_dict[k] for k in requir_feature_name}
            category_features_list.append(features_dict)
            # text feature
            doc = features[-1].strip()
            doc = doc.replace('\t', ' , ')
            doc = doc.split(' ')
            raw_doc_list.append(doc)
            doc = [self.word_dict.get(word, 0) for word in doc]
            docs_list.append(doc)
        id_to_row = {k: v for v, k in enumerate(ids_list)}
        docs_matrix = np.array(docs_list)
        if one_hot:
            vec = feature_extraction.DictVectorizer()
            category_features_matrix = vec.fit_transform(category_features_list)
            category_features_name = vec.get_feature_names()
            return id_to_row, category_features_list, category_features_matrix, category_features_name, docs_matrix, raw_doc_list
        else:
            return id_to_row, category_features_list, docs_matrix, raw_doc_list

    def feature_lookup(self, idstr, idtype, raw=0):
        if idtype == 'expect':
            row = self.exp_to_row[idstr]
            features = self.exp_features[row]
            if raw:
                docs = self.exp_doc_raw[row]
            else:
                docs = self.exp_docs[row][:self.doc_len]
        else:
            row = self.job_to_row[idstr]
            features = self.job_features[row]
            if raw:
                docs = self.job_doc_raw[row]
            else:
                docs = self.job_docs[row][:self.doc_len]
        if len(docs) < self.doc_len:
            if raw:
                docs.extend([' ']*(self.doc_len - len(docs)))
            else:
                docs.extend([0]*(self.doc_len - len(docs)))
        return [features, docs]

    def data_generator(self, fp, batch_size, raw=0):
        with open(fp) as f:
            batch_data = []
            for line in f:
                eid, jid, label = line.strip().split('\001')
                label = int(label)
                if jid not in self.job_to_row or eid not in self.exp_to_row:
                    print('loss id')
                    continue
                job_data, jd = self.feature_lookup(jid, 'job', raw)
                exp_data, cv = self.feature_lookup(eid, 'expect', raw)
                cate_features = sparse.hstack([job_data, exp_data])
                cate_features = cate_features.col
                if raw:
                    cate_features = [self.feature_name_sparse[i] for i in cate_features]
                    batch_data.append([jid, eid, jd, cv, cate_features, label])
                else:
                    batch_data.append([jd, cv, cate_features, label])
                if len(batch_data) >= batch_size:
                    if raw:
                        yield batch_data
                    else:
                        batch_data = list(zip(*batch_data))
                        yield [np.array(x) for x in batch_data]
                    batch_data = []


if __name__ == '__main__':
    mix_data = MixData(
        '../Data/multi_data/multi_data',
        './data/multi_data',
        5,
        100,
    )
    fp = './data/multi_data.train'
    g = mix_data.data_generator(fp, 2)
    print(next(g))
    print("done")