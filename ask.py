import torch
from sqlnet.utils import *
from sqlnet.model.sqlnet import SQLNet
from sqlnet.query import Query
import nltk



def best_model_name(train_emb, ca, for_load=False):
    new_data = 'old'
    mode = 'sqlnet'
    if for_load:
        use_emb = use_rl = ''
    else:
        use_emb = '_train_emb' if train_emb else ''
        use_rl = ''
    use_ca = '_ca' if ca else ''

    agg_model_name = 'saved_model/%s_%s%s%s.agg_model'%(new_data,
            mode, use_emb, use_ca)
    sel_model_name = 'saved_model/%s_%s%s%s.sel_model'%(new_data,
            mode, use_emb, use_ca)
    cond_model_name = 'saved_model/%s_%s%s%s.cond_%smodel'%(new_data,
            mode, use_emb, use_ca, use_rl)

    if not for_load and train_emb:
        agg_embed_name = 'saved_model/%s_%s%s%s.agg_embed'%(new_data,
                mode, use_emb, use_ca)
        sel_embed_name = 'saved_model/%s_%s%s%s.sel_embed'%(new_data,
                mode, use_emb, use_ca)
        cond_embed_name = 'saved_model/%s_%s%s%s.cond_embed'%(new_data,
                mode, use_emb, use_ca)

        return agg_model_name, sel_model_name, cond_model_name,\
                agg_embed_name, sel_embed_name, cond_embed_name
    else:
        return agg_model_name, sel_model_name, cond_model_name


def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql['question_tok'])
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header']))
        """
        ans_seq.append((sql['sql']['agg'],
            sql['sql']['sel'],
            len(sql['sql']['conds']),
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds'])))
        query_seq.append(sql['query_tok'])
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'],
            table_data[sql['table_id']]['header'], sql['query']))
        """
        ans_seq.append(())
        query_seq.append("")
        gt_cond_seq.append([])
        vis_seq.append(())

    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq


def ask(model, sql_data, pred_entry):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    batch_size = 2
    table_data = model.val_table_data

    ed = st+batch_size if st+batch_size < len(perm) else len(perm)

    q_seq, col_seq, col_num, _, _, _, raw_data = to_batch_seq([sql_data, sql_data], table_data, perm, st, ed, ret_vis_data=True)
    raw_q_seq = [sql_data["question"], sql_data["question"]]

    gt_sel_seq = [0, 0] # TODO: not sure what this does
    score = model.forward(q_seq, col_seq, col_num,
            pred_entry, gt_sel = gt_sel_seq)
    pred_queries = model.gen_query(score, q_seq, col_seq,
            raw_q_seq, None, pred_entry)

    return pred_queries[0]


def load_model():
    train_emb = True
    ca = True

    N_word=300
    B_word=42
    USE_SMALL=False
    GPU=True

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
        load_used=True, use_small=USE_SMALL) # load_used can speed up loading

    model = SQLNet(word_emb, N_word=N_word, use_ca=ca, gpu=GPU,
            trainable_emb = True)

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
                    0, use_small=USE_SMALL)

    model.val_table_data = val_table_data

    if train_emb:
        agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(train_emb, ca)
        print "Loading from %s"%agg_m
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print "Loading from %s"%sel_m
        model.sel_pred.load_state_dict(torch.load(sel_m))
        print "Loading from %s"%cond_m
        model.cond_pred.load_state_dict(torch.load(cond_m))
        print "Loading from %s"%agg_e
        model.agg_embed_layer.load_state_dict(torch.load(agg_e))
        print "Loading from %s"%sel_e
        model.sel_embed_layer.load_state_dict(torch.load(sel_e))
        print "Loading from %s"%cond_e
        model.cond_embed_layer.load_state_dict(torch.load(cond_e))
    else:
        agg_m, sel_m, cond_m = best_model_name(train_emb, ca)
        print "Loading from %s"%agg_m
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print "Loading from %s"%sel_m
        model.sel_pred.load_state_dict(torch.load(sel_m))
        print "Loading from %s"%cond_m
        model.cond_pred.load_state_dict(torch.load(cond_m))

    return model


def ask_question(model, question, table_id):
    TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)

    question = {"question": question,
                "table_id": table_id,
                "question_tok": [t.lower() for t in nltk.word_tokenize(question)]}

    query = ask(model, question, TEST_ENTRY)
    query = Query(query["sel"], query["agg"], query["conds"])
    return query


if __name__ == "__main__":
    model = load_model()
    print(ask_question(model, 'What position does the player who played for butler cc (ks) play?', '1-10015132-11'))
