"""Optional local Qwen3 reranking. Does not read qrels or narrative fields.

Keep original rankings intact and rerank only the top candidate set.
Model prompt/scoring follows Qwen's official Transformers example.
"""
from __future__ import annotations
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from collections import Counter
from importlib.metadata import version

MODEL_ID = 'Qwen/Qwen3-Reranker-0.6B'
INSTRUCTION = ('Given a university administrative information need, retrieve regulations, '
               'instructions or forms that directly help answer it. Judge relevance, not approval of a case.')
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
          'and the Instruct provided. Note that the answer can only be "yes" or "no".'
          '<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
WORD = re.compile(r'[^\W_]+', re.UNICODE)


def read_run(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    rows, seen = [], set()
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError('Run must contain six TREC columns per line')
            qid, _, docno, rank, score, tag = parts
            if (qid, docno) in seen:
                raise ValueError(f'Duplicate document {docno} for query {qid}')
            seen.add((qid, docno))
            number = float(score)
            if not math.isfinite(number):
                raise ValueError('Non-finite first-stage score')
            rows.append(dict(qid=qid, docno=docno, rank=int(rank), score=number))
    if not rows:
        raise ValueError('Input ranking is empty')
    return rows


def group_run(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['qid'], []).append(row)
    for hits in groups.values():
        hits.sort(key=lambda r: (r['rank'], -r['score']))
    return groups


def select_passages(text, query, window=200, stride=160, count=2, char_limit=200_000):
    """Bounded lexical passage selection; heuristic, not a learned selector.

    Include a short lead plus highest-overlap windows, preserving source order.
    No dictionary based on the evaluation queries is used.
    """
    words = (text or '')[:char_limit].split()
    if len(words) <= window:
        return ' '.join(words), [0]
    wanted = {t.casefold() for t in WORD.findall(query) if len(t) >= 3}
    windows = [(i, words[i:i+window]) for i in range(0, len(words), stride)]
    def relevance(item):
        counts = Counter(t.casefold() for t in WORD.findall(' '.join(item[1])))
        return sum(1 + math.log(counts[t]) for t in wanted if counts[t])
    chosen = sorted(sorted(windows, key=lambda x: (-relevance(x), x[0]))[:count])
    lead = ' '.join(words[:40])
    content = '\n\n'.join(' '.join(w) for _, w in chosen)
    return lead + '\n\n' + content, [i for i, _ in chosen]


def reorder(hits, scores, depth):
    """Rerank top depth; keep tail order and document membership exactly.

    Stored scores encode final rank; model scores are saved separately.
    """
    head, tail = hits[:depth], hits[depth:]
    if len(scores) != len(head) or any(not math.isfinite(s) for s in scores):
        raise ValueError('Need one finite model score for every candidate')
    ordered = [pair[1] for pair in sorted(enumerate(head), key=lambda x: (-scores[x[0]], x[0]))]
    ordered += tail
    return [{**row, 'rank': rank, 'score': float(len(ordered)-rank)} for rank, row in enumerate(ordered)]


class QwenScorer:
    def __init__(self, model_path, device='cpu', max_length=1536, batch_size=1, threads=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if max_length < 256 or max_length > 32768:
            raise ValueError('max_length must be in [256,32768]')
        if batch_size < 1 or threads < 1:
            raise ValueError('Batch size and threads must be positive')
        torch.set_num_threads(threads)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested, but CUDA is unavailable')
        self.torch, self.device = torch, device
        self.batch_size, self.max_length = batch_size, max_length
        # Runtime is intentionally offline. Download/tokenizer prep is a separate step.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        ).to(device).eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.yes = self.tokenizer.convert_tokens_to_ids('yes')
        self.no = self.tokenizer.convert_tokens_to_ids('no')
        if self.yes == self.no or self.yes is None or self.no is None:
            raise ValueError('Expected distinct yes/no tokens for Qwen3 reranker')

    def score(self, query, documents):
        scores, usage = [], []
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)[:256]
        query = self.tokenizer.decode(query_ids)
        instruction = f'<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: '
        query_prompt = self.tokenizer.encode(instruction, add_special_tokens=False)
        available = self.max_length - len(self.prefix) - len(query_prompt) - len(self.suffix)
        if available <= 0:
            raise ValueError('No space remains for document text')
        for start in range(0, len(documents), self.batch_size):
            sequences = []
            for doc in documents[start:start+self.batch_size]:
                ids = self.tokenizer.encode(doc, add_special_tokens=False)[:available]
                sequence = self.prefix + query_prompt + ids + self.suffix
                sequences.append(sequence)
                usage.append(len(sequence))
            inputs = self.tokenizer.pad({'input_ids': sequences}, padding=True, return_tensors='pt')
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                # Qwen3 supports keeping only the last-position logits, saving memory.
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                margins = (logits[:, self.yes] - logits[:, self.no]).cpu().tolist()
            scores.extend(float(x) for x in margins)
        return scores, usage


def rerank_dataset(dataset, output, model_path, depth=20, device='cpu', max_length=1536,
                   batch_size=1, threads=4):
    """Apply stage two to output/run.txt.gz; preserve first-stage artifacts."""
    from tira.third_party_integrations import ir_datasets
    from components import EventLog, TRACE_FILE
    output = Path(output)
    if depth < 1:
        raise ValueError('depth must be positive')
    original = output / 'run.first-stage.txt.gz'
    run_path = output / 'run.txt.gz'
    # Existing backup identifies this experiment's source even when rerunning stage two.
    source = original if original.exists() else run_path
    groups = group_run(read_run(source))
    ds = ir_datasets.load(dataset)
    queries = {}
    for q in ds.queries_iter():
        metadata = q.original_query if isinstance(q.original_query, dict) else {}
        description = metadata.get('description', '')
        if not isinstance(description, str):
            description = ''
        queries[str(q.query_id)] = q.default_text() + '\n' + description
    unknown = set(groups) - queries.keys()
    if unknown:
        raise ValueError(f'Run has unknown query IDs: {sorted(unknown)}')
    wanted = {r['docno'] for hits in groups.values() for r in hits[:depth]}
    texts = {}
    for doc in ds.docs_iter():
        docno = str(doc.doc_id)
        if docno in wanted:
            title = getattr(doc, 'title', '')
            title = title if isinstance(title, str) else ''
            texts[docno] = title + '\n' + (doc.default_text() or '')
    missing = wanted - texts.keys()
    if missing:
        raise ValueError(f'Candidate documents missing from dataset: {sorted(missing)[:3]}')
    scorer = QwenScorer(model_path, device, max_length, batch_size, threads)
    start = time.perf_counter()
    base_trace = output / 'run-trace.first-stage.jsonl.log.gz'
    trace_path = output / TRACE_FILE
    trace_source = base_trace if base_trace.exists() else trace_path
    parents, highest = {}, 0
    if trace_source.exists():
        with gzip.open(trace_source, 'rt', encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                parents[str(event['case_id'])] = event['event_id']
                try:
                    highest = max(highest, int(event['event_id'].split('-')[-1]))
                except ValueError:
                    raise ValueError('Unsupported first-stage event ID format')
    if set(groups) - parents.keys():
        raise ValueError('Missing first-stage trace chains for ranked queries')
    output_rows, details = [], []
    temp_trace = output / 'rerank-events.tmp.gz'
    try:
        with EventLog(temp_trace, model=MODEL_ID) as log:
            log._count = highest
            for qid, hits in groups.items():
                candidates, passages = [], []
                for hit in hits[:depth]:
                    passage, positions = select_passages(texts[hit['docno']], queries[qid])
                    passages.append(passage)
                    candidates.append({'docno':hit['docno'], 'window_word_offsets':positions})
                parent = parents[qid]
                try:
                    scores, usage = scorer.score(queries[qid], passages)
                except Exception as exc:
                    log.log(qid, 'model_call', parent=parent, input={'candidates':candidates},
                            status='error', error=str(exc))
                    raise
                output_rows.extend(reorder(hits, scores, depth))
                records = [{**c, 'yes_no_logit_margin':s, 'input_tokens':u}
                           for c,s,u in zip(candidates,scores,usage)]
                details.append({'qid':qid, 'candidates':records})
                parent = log.log(qid, 'model_call', parent=parent,
                                 input={'query':queries[qid], 'instruction':INSTRUCTION,
                                        'candidate_selection':'top first-stage ranks', 'candidates':candidates,
                                        'max_length':max_length}, output={'scores':records})
                top = sorted(zip(candidates,scores), key=lambda x:-x[1])[:10]
                log.log(qid, 'decision', parent=parent, output={
                    'decision':'ranking', 'answer':[c['docno'] for c,s in top],
                    'evidence':[{'source_id':c['docno'], 'relevance':'Qwen3 relevance margin', 'score':s} for c,s in top],
                    'ranked_documents':len(hits)})
        # Prepare all outputs first. Failed scoring leaves original rankings untouched.
        fd, name = tempfile.mkstemp(dir=output, suffix='.txt.gz')
        os.close(fd)
        temp_run = Path(name)
        with gzip.open(temp_run, 'wt', encoding='utf-8') as stream:
            for row in output_rows:
                stream.write(f"{row['qid']} Q0 {row['docno']} {row['rank']} {row['score']:.9g} qwen3-reranked\n")
        if not original.exists():
            original.write_bytes(run_path.read_bytes())
        if not base_trace.exists():
            base_trace.write_bytes(trace_path.read_bytes())
        # Concatenated gzip members are valid gzip; all original event IDs are preserved.
        new_trace = output / 'trace.complete.tmp.gz'
        new_trace.write_bytes(base_trace.read_bytes() + temp_trace.read_bytes())
        os.replace(new_trace, trace_path)
        temp_run.chmod(0o644)
        os.replace(temp_run, run_path)
        (output / 'rerank_scores.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
        cfg_path = output / 'retrieval_config.json'
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        if 'first_stage_config' not in cfg:
            cfg = {'first_stage_config':cfg}
        cfg.update({'approach':'collection-routing+qwen3-reranking', 'uses_neural_reranker':True,
                    'uses_llm':True, 'uses_embeddings':False, 'uses_relevance_judgments':False,
                    'reranker':MODEL_ID, 'model_path':str(model_path), 'device':scorer.device,
                    'depth':depth, 'max_length':max_length, 'batch_size':batch_size,
                    'score_output':'rank-derived; raw model margins in rerank_scores.json',
                    'passage_selection':'40-word lead + 2 highest lexical-overlap 200-word windows; first 200k chars',
                    'seconds_excluding_model_load':round(time.perf_counter()-start,3),
                    'torch_version':version('torch'), 'transformers_version':version('transformers')})
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        if temp_trace.exists():
            os.replace(temp_trace, output / 'rerank-error-trace.jsonl.log.gz')
        raise
    finally:
        temp_trace.unlink(missing_ok=True)
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rerank an existing first-stage run; no re-indexing')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--device', choices=['cpu','cuda','auto'], default='cpu')
    parser.add_argument('--max-length', type=int, default=1536)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(rerank_dataset(args.dataset,args.output,args.model_path,args.depth,
                                   args.device,args.max_length,args.batch_size,args.threads),indent=2))
