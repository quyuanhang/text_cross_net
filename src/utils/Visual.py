import tensorflow as tf
from tqdm import tqdm
from nets.TextCrossNet import TextCrossNet
from utils.Trainer import feed_dict


def feature_fold(idx, doc_len, block_len):
    idx_fold = []
    for i in idx:
        i_of_mat = i % (doc_len ** 2)
        i_of_feature = i // (doc_len ** 2)
        row = i_of_mat // doc_len
        col = i_of_mat % doc_len
        idx_fold.append([
            row - block_len // 2,
            row + block_len // 2,
            col - block_len // 2,
            col + block_len // 2,
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


def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def mk_html(doc, attns):
    html = ""
    for word, attn in zip(doc, attns):
        html += ' ' + highlight(
            word,
            attn,
        )
    return html + "<br><br>\n"


def feature_lookup(predictions, idxs, datas):
    visual_str = ""
    for predict, idx, data in zip(predictions, idxs, datas):
        if round(predict[0]) == 0:
            continue
        jid, eid, jd, cv, cate_features, label = data
        jd_attn = [0] * len(jd)
        cv_attn = [0] * len(cv)
        if label == 0:
            continue
        for jd_idx1, jd_idx2, cv_idx1, cv_idx2, cate_idx in idx:
            jd_attn[jd_idx1: jd_idx2+1] = [0.5] * (jd_idx2 - jd_idx1)
            cv_attn[cv_idx1: cv_idx2+1] = [0.5] * (cv_idx2 - cv_idx1)
        jd = mk_html(jd, jd_attn)
        cv = mk_html(cv, cv_attn)
        visual_str += "<p> jobid: {} expectid: {} </p>".format(jid, eid)
        visual_str += jd + '\n' + cv + '\n'
        visual_str += "<p>==============================================</p>"
    return visual_str


# def feature_lookup(predictions, idxs, datas):
#     visual_str = ""
#     for predict, idx, data in zip(predictions, idxs, datas):
#         if round(predict[0]) == 0:
#             continue
#         jid, eid, jd, cv, cate_features, label = data
#         if label == 0:
#             continue
#         feature_str = ""
#         for jd_idx1, jd_idx2, cv_idx1, cv_idx2, cate_idx in idx:
#             jd_idx1 = find(jd, jd_idx1, (',', ' '), 'backward')
#             jd_idx2 = find(jd, jd_idx2, (',', ' '), 'forward')
#             cv_idx1 = find(cv, cv_idx1, (',', ' '), 'backward')
#             cv_idx2 = find(cv, cv_idx2, (',', ' '), 'forward')
#             cate_feature = cate_features[cate_idx]
#             jd_block = "".join(jd[jd_idx1+1: jd_idx2])
#             cv_block = "".join(cv[cv_idx1+1: cv_idx2])
#             feature_str += "【{}: {} -> {}】\n".format(cate_feature, jd_block, cv_block)
#         feature_str = "jobid:{} expectid:{}\n{}\n{}\n{}\n{}\n".format(
#             jid,
#             eid,
#             feature_str,
#             ",".join(cate_features),
#             "".join(jd),
#             "".join(cv),
#         )
#         feature_str += "\n==========================\n"
#         visual_str += feature_str
#     return visual_str


def visual(
        sess: tf.Session,
        model: TextCrossNet,
        test_data_fn,
        raw_data_fn,
        data_len=1,
    ):

    predict = model.predict
    related_features = model.related_feature
    test_data = test_data_fn()
    raw_data = raw_data_fn()
    visual_str = ""
    for i, batch in tqdm(enumerate(test_data)):
        if i > data_len:
            break
        fd = feed_dict(batch)
        predict_data, feature_indexs = sess.run([predict, related_features], feed_dict=fd)
        x = 1
        feature_indexs = [
            feature_fold(idx, model.doc_len, model.block_len)
            for idx in feature_indexs]
        batch_raw = next(raw_data)
        batch_visual_str = feature_lookup(predict_data, feature_indexs, batch_raw)
        visual_str += batch_visual_str
    return visual_str



