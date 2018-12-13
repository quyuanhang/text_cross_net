import numpy as np
import tensorflow as tf
from nets.TextCrossNet import TextCrossNet
from utils.MixData import MixData
from tqdm import tqdm
from sklearn import metrics


def feed_dict(data):
    jds, cvs, cate_features, labels = data
    fd = {
        'jds:0': jds,
        'cvs:0': cvs,
        'cate_features:0': cate_features,
        'loss/labels:0': labels
    }
    return fd


def train(
        sess: tf.Session,
        model: TextCrossNet,
        writer,
        train_data_fn,
        test_data_fn,
        lr=0.0005,
        n_epoch=100,
    ):

    predict = model.predict

    loss = model.loss

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        epoch_loss = []
        epoch_metric_outside = []
        train_data = train_data_fn()
        for batch in tqdm(train_data, ncols=50):
            fd = feed_dict(batch)
            predict_data, loss_data, _ = sess.run(
                [predict, loss, train_op],
                feed_dict=fd)
            epoch_loss.append(loss_data)
            outside_auc_data = metrics.roc_auc_score(fd['loss/labels:0'], predict_data)
            epoch_metric_outside.append(outside_auc_data)
        val_loss = []
        val_metric_outside = []
        test_data = test_data_fn()
        for batch in tqdm(test_data):
            fd = feed_dict(batch)
            predict_data, loss_data = sess.run(
                [predict, loss],
                feed_dict=fd)
            val_loss.append(loss_data)
            outside_auc_data = metrics.roc_auc_score(fd['loss/labels:0'], predict_data)
            val_metric_outside.append(outside_auc_data)
        print(
            'epoch: {}\n'.format(epoch),
            'train loss: {:.3f} train metric: {:.3f}\n'.format(
                np.array(epoch_loss).mean(),
                np.array(epoch_metric_outside).mean()),
            'valid loss: {:.3f} valid metric: {:.3f}\n'.format(
                np.array(val_loss).mean(),
                np.array(val_metric_outside).mean()),
        )


def feature_fold(idx, n_feature, doc_len, block_len):
    idx_fold = []
    for i in idx:
        i_of_mat = i % n_feature
        i_of_feature = i // (doc_len ** 2)
        row = i_of_mat // doc_len
        col = i_of_mat % doc_len
        idx_fold.append([
            row,
            row + block_len,
            col,
            col + block_len,
            i_of_feature
        ])
    return idx_fold


def find(l: list, i: int, items, mode):
    if mode == 'forward':
        for i in range(i, len(l)):
            if l[i] in items:
                return i
    else:
        for i in range(i, -1, -1):
            if [i] in items:
                return i
    return i


def feature_lookup(predictions, idxs, datas):
    visual_str = ""
    for predict, idx, data in zip(predictions, idxs, datas):
        if round(predict[0]) == 0:
            continue
        jid, eid, jd, cv, cate_features, label = data
        if label == 0:
            continue
        feature_str = ""
        for jd_idx1, jd_idx2, cv_idx1, cv_idx2, cate_idx in idx:
            jd_idx1 = find(jd, jd_idx1, (',', ' '), 'backward')
            jd_idx2 = find(jd, jd_idx2, (',', ' '), 'forward')
            cv_idx1 = find(cv, cv_idx1, (',', ' '), 'backward')
            cv_idx2 = find(cv, cv_idx2, (',', ' '), 'forward')
            cate_feature = cate_features[cate_idx]
            jd_block = "".join(jd[jd_idx1+1: jd_idx2])
            cv_block = "".join(cv[cv_idx1+1: cv_idx2])
            feature_str += "【{}: {} -> {}】\n".format(cate_feature, jd_block, cv_block)
        feature_str = "jobid:{} expectid:{}\n{}\n{}\n{}\n{}\n".format(
            jid,
            eid,
            feature_str,
            ",".join(cate_features),
            "".join(jd),
            "".join(cv),
        )
        feature_str += "\n==========================\n"
        visual_str += feature_str
    return visual_str


def visual(
        sess: tf.Session,
        model: TextCrossNet,
        test_data_fn,
        raw_data_fn,
    ):

    predict = model.predict
    related_features = model.related_feature
    test_data = test_data_fn()
    raw_data = raw_data_fn()
    visual_str = ""
    for batch in tqdm(test_data):
        fd = feed_dict(batch)
        predict_data, feature_indexs = sess.run([predict, related_features], feed_dict=fd)
        feature_indexs = [
            feature_fold(idx, model.feature_len, model.doc_len, model.block_len)
            for idx in feature_indexs]
        batch_raw = next(raw_data)
        batch_visual_str = feature_lookup(predict_data, feature_indexs, batch_raw)
        visual_str += batch_visual_str
    return visual_str




