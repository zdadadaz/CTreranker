

def write_hits(hits, output_path, excludeZero = False, run_name='noName'):
    trec = []
    # qid
    for qid in hits.keys():
        #docid
        cnt = 1
        for hit in hits[qid]:
            if (not excludeZero) or (hit.score > 0.0001):
                trec.append(str(qid) + "\tQ0\t"+str(hit.docid) + "\t"+ str(cnt) +"\t"+str(hit.score)+ "\t" +run_name+"\n")
                cnt+=1
    trec.sort(key=lambda k: int(k.split("\t")[0]))
    with open(output_path+'.res', "w") as f:
        f.writelines(trec)

def write_rerank_res(org, rerank, k, run_name, output_path):
    trec = []
    # qid
    for qid in rerank.keys():
        # docid
        for kid in range(k):
            trec.append(
                str(qid) + "\tQ0\t" + rerank[qid]['docid'][kid] + "\t" + rerank[qid]['rank'][kid] + "\t" + rerank[qid]['score'][kid] + "\t" + run_name + "\n")
        for kid in range(k,len(org[qid]['docid'])):
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

