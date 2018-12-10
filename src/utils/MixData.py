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
        expect_features_names = [
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
        self.word_dict = self.build_dict(fps, wfreq)
        self.exp_to_row, self.exp_features, self.exp_features_names, self.exp_docs, self.exp_doc_raw = \
            self.build_features('{}.profile.expect'.format(fpin), expect_features_names)
        self.job_to_row, self.job_features, self.job_features_names, self.job_docs, self.job_doc_raw = \
            self.build_features('{}.profile.job'.format(fpin), job_features_names)

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
        word_dict = {k: v for k, v in words_freq.items() if v >= w_freq}
        word_dict = {k: v for v, k in enumerate(word_dict.keys(), 2)}
        word_dict['__pad__'] = 0
        word_dict['__unk__'] = 1
        print('n_words: {}'.format(len(word_dict)))
        return word_dict

    def build_features(self, fp, feature_name, one_hot=True):
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
            doc = [self.word_dict.get(word, 0) for word in doc]
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

    def feature_lookup(self, idstr, idtype):
        if idtype == 'expect':
            row = self.exp_to_row[idstr]
            features = self.exp_features[row]
            docs = self.exp_docs[row][:self.doc_len]
        else:
            row = self.job_to_row[idstr]
            features = self.job_features[row]
            docs = self.job_docs[row][:self.doc_len]
        if len(docs) < self.doc_len:
            docs.extend([0]*(self.doc_len - len(docs)))
        return [features, docs]

    def data_generator(self, fp, batch_size):
        batch_data = []
        with open(fp) as f:
            end = False
            while not end:
                while len(batch_data) < batch_size:
                    line = f.readline()
                    if not line:
                        end = True
                        break
                    # jid, eid, label = line.strip().split('\001')
                    eid, jid, label = line.strip().split('\001')
                    label = int(label)
                    if jid not in self.job_to_row or eid not in self.exp_to_row:
                        # print('loss id')
                        continue
                    job_data, jd = self.feature_lookup(jid, 'job')
                    exp_data, cv = self.feature_lookup(eid, 'expect')
                    cate_features = sparse.vstack(job_data, exp_data)
                    batch_data.append([jd, cv, cate_features, label])
                # yield batch_data
                yield list(zip(*batch_data))

if __name__ == '__main__':
    mix_data = MixData(
        '../Data/multi_data/multi_data',
        './data/multi_data',
        5,
        100,
    )
    fp = './data/multi_data.train'
    g = mix_data.data_generator(fp, 10)