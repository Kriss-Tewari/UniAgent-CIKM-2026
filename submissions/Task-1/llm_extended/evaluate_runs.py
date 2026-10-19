"""Compare unchanged reference and reranked run against the SAME local qrels."""
import argparse
import gzip
import ir_measures
parser=argparse.ArgumentParser()
parser.add_argument('--qrels',required=True)
parser.add_argument('--baseline',required=True)
parser.add_argument('--reranked',required=True)
args=parser.parse_args()
def lines(path):
    return (gzip.open if path.endswith('.gz') else open)(path,'rt',encoding='utf-8')
with lines(args.qrels) as f:
    qrels=list(ir_measures.read_trec_qrels(f))
measures=[ir_measures.nDCG@10,ir_measures.AP,ir_measures.R@100,ir_measures.RR@10]
for label,path in [('BASELINE',args.baseline),('RERANKED',args.reranked)]:
    with lines(path) as f:
        run=list(ir_measures.read_trec_run(f))
    print(label,ir_measures.calc_aggregate(measures,qrels,run))
