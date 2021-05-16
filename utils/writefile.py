
def write_hits(hits, output_path, bm25_k=1000, excludeZero = False, run_name='noName'):
    trec = []
    # qid
    for qid in hits.keys():
        # docid
        cnt = 1
        for hit in hits[qid]:
            if cnt > bm25_k :
                break
            if (not excludeZero) or (hit.score > 0.0001):
                trec.append(str(qid) + "\tQ0\t" +str(hit.docid) + "\t" + str(cnt) +"\t" + str(
                    hit.score) + "\t" + run_name + "\n")
                cnt += 1
    trec.sort(key=lambda k: int(k.split("\t")[0]))
    with open(output_path + '.res', "w") as f:
        f.writelines(trec)


def write_rerank_res(org, rerank, k, run_name, output_path):
    trec = []
    # qid
    for qid in rerank.keys():
        # docid
        for kid in range(k):
            trec.append(
                str(qid) + "\tQ0\t" + rerank[qid]['docid'][kid] + "\t" + rerank[qid]['rank'][kid] + "\t" +
                str(float(rerank[qid]['score'][kid])+float(org[qid]['score'][k])) + "\t" + run_name + "\n")
        for kid in range(k, len(org[qid]['docid'])):
            trec.append(
                str(qid) + "\tQ0\t" + org[qid]['docid'][kid] + "\t" + org[qid]['rank'][kid] + "\t" +
                org[qid]['score'][kid] + "\t" + run_name + "\n")

    trec.sort(key=lambda k: int(k.split("\t")[0]))
    with open(output_path + '.res', "w") as f:
        f.writelines(trec)


def write_qrels(qids, qrels, out_path):
    qrel_out = []
    for qid in qids:
        qid = str(qid)
        for doc in qrels[qid]:
            if doc[:3] == 'NCT':
                text = qid + " 0 " + doc + " " + str(qrels[qid][doc]) + "\n"
                qrel_out.append(text)

    qrel_out.sort(key=lambda k: int(k.split(" ")[0]))
    with open(out_path, "w") as f:
        f.writelines(qrel_out)


def write_eval(eval, out_path):
    out = []
    for m in eval.keys():
        text = m + "\t" + str(eval[m]) + "\n"
        out.append(text)

    with open(out_path + '.eval', "w") as f:
        f.writelines(out)


def write_pair(a, b, out_path):
    out = []
    for i in range(len(a)):
        out.append(str(a[i]) + ' ' + str(b[i]) + '\n')
    # out.sort(key=lambda k: int(k.split(" ")[0].split("_")[0]))
    with open(out_path, "w") as f:
        f.writelines(out)


def write_pos_qrels(qrels, qids, out_path):
    qrel_out = []
    for qid in qrels.keys():
        if qid in qids:
            for doc in qrels[qid]:
                if doc[:3] == 'NCT' and int(qrels[qid][doc]) > 0:
                    text = qid + " 0 " + doc + " " + str(qrels[qid][doc]) + "\n"
                    qrel_out.append(text)

    qrel_out.sort(key=lambda k: int(k.split(" ")[0]))
    with open(out_path, "w") as f:
        f.writelines(qrel_out)


def write_queries(queris, qids, out_path):
    qrel_out = []
    for qid in queris.keys():
        if qid in qids:
            text = str(qid) + "\t" + queris[qid]["text"] + "\t" + str(queris[qid]["age"]) + "\t" + str(
                queris[qid]["gender"]) + "\n"
            qrel_out.append(text)

    qrel_out.sort(key=lambda k: int(k.split("\t")[0]))
    with open(out_path, "w") as f:
        f.writelines(qrel_out)


def write_res(res, output_path):
    trec = []
    for qid in res.keys():
        n = len(res[qid]['score'])
        for i in range(n):
            trec.append(
                str(qid) + "\tQ0\t" + res[qid]['docid'][i] + "\t" + str(res[qid]['rank'][i]) + "\t" + str(res[qid]['score'][
                    i]) + "\t" + 'noName' + "\n")

    trec.sort(key=lambda k: int(k.split("\t")[0]))
    with open(output_path + '.res', "w") as f:
        f.writelines(trec)
