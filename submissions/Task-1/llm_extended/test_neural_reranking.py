"""Model-free tests: ranking invariants and full stage-two orchestration.
These do NOT establish actual neural inference or retrieval effectiveness.
"""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import neural_reranking as n

class FakeLog:
    def __init__(self,path,model='-'):
        self.f=gzip.open(path,'wt',encoding='utf-8'); self._count=0; self.model=model
    def __enter__(self): return self
    def __exit__(self,*args): self.f.close()
    def log(self,qid,kind,parent=None,**kwargs):
        self._count+=1; eid=f'evt-{self._count:04d}'
        self.f.write(json.dumps(dict(case_id=qid,event_id=eid,parent_event_id=parent,
                                    event_type=kind,model=self.model,**kwargs))+'\n')
        return eid

class FakeScorer:
    device='cpu'
    def __init__(self,*a): pass
    def score(self,query,documents): return [0.1,3.0][:len(documents)], [50]*len(documents)

class Tests(unittest.TestCase):
    def test_reorder_preserves_tail_membership_and_stable_ties(self):
        rows=[dict(qid='1',docno=str(i),rank=i,score=10-i) for i in range(5)]
        result=n.reorder(rows,[0.,2.,2.],3)
        self.assertEqual([r['docno'] for r in result],['1','2','0','3','4'])
        self.assertEqual([r['rank'] for r in result],list(range(5)))
        self.assertTrue(all(a['score']>b['score'] for a,b in zip(result,result[1:])))
        self.assertEqual({r['docno'] for r in result},{r['docno'] for r in rows})
        with self.assertRaises(ValueError): n.reorder(rows,[float('nan')],1)

    def test_passage_selector_can_find_late_evidence(self):
        words=['boilerplate']*480+['hotel','reimbursement']*100
        selected,offsets=n.select_passages(' '.join(words),'hotel reimbursement',count=1)
        self.assertIn('reimbursement',selected)
        self.assertGreater(offsets[0],0)
        self.assertEqual(n.select_passages('', 'q'),('',[0]))

    def test_duplicate_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d,'run.txt'); p.write_text('1 Q0 a 0 2 run\n1 Q0 a 1 1 run\n')
            with self.assertRaises(ValueError): n.read_run(p)

    def test_full_stage_preserves_original_and_extends_trace(self):
        query=types.SimpleNamespace(query_id='1',original_query={'description':'hotel costs',
                  'narrative':'SECRET'},default_text=lambda:'Travel')
        docs=[types.SimpleNamespace(doc_id=c,default_text=lambda c=c:'Travel hotel '+c) for c in 'abc']
        ds=types.SimpleNamespace(queries_iter=lambda:iter([query]),docs_iter=lambda:iter(docs))
        tira=types.ModuleType('tira'); third=types.ModuleType('tira.third_party_integrations')
        third.ir_datasets=types.SimpleNamespace(load=lambda _:ds)
        components=types.ModuleType('components'); components.EventLog=FakeLog
        components.TRACE_FILE='run-trace.jsonl.log.gz'
        modules={'tira':tira,'tira.third_party_integrations':third,'components':components}
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules,modules), patch.object(n,'QwenScorer',FakeScorer), patch.object(n,'version',lambda _: 'test-double'):
            output=Path(d)
            with gzip.open(output/'run.txt.gz','wt') as f:
                f.write('1 Q0 a 0 5 run\n1 Q0 b 1 4 run\n1 Q0 c 2 3 run\n')
            with gzip.open(output/components.TRACE_FILE,'wt') as f:
                f.write(json.dumps(dict(case_id='1',event_id='evt-0001',parent_event_id=None,event_type='decision'))+'\n')
            before=(output/'run.txt.gz').read_bytes()
            n.rerank_dataset('dataset',output,'local-model',depth=2)
            self.assertEqual((output/'run.first-stage.txt.gz').read_bytes(),before)
            self.assertEqual([r['docno'] for r in n.read_run(output/'run.txt.gz')],['b','a','c'])
            with gzip.open(output/components.TRACE_FILE,'rt') as f: events=[json.loads(x) for x in f]
            self.assertEqual([e['event_id'] for e in events],['evt-0001','evt-0002','evt-0003'])
            self.assertEqual(events[1]['parent_event_id'],'evt-0001')
            self.assertNotIn('SECRET',json.dumps(events))
            stable=(output/'run.txt.gz').read_bytes()
            def fail(*a): raise RuntimeError('simulated model failure')
            with patch.object(FakeScorer,'score',fail):
                with self.assertRaises(RuntimeError): n.rerank_dataset('dataset',output,'local-model',depth=2)
            self.assertEqual((output/'run.txt.gz').read_bytes(),stable)
            self.assertTrue((output/'rerank-error-trace.jsonl.log.gz').exists())

if __name__=='__main__': unittest.main()
