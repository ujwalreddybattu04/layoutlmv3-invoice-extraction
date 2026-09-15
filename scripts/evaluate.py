"""Measure exact field values and box IoU; retain every actual output and trace."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.aggregate import aggregate
from src.merge import intersection_over_union
from src.pipeline import Pipeline
from src.types import Prediction, Word
from src.visualize import annotate
from src.ocr import load_image


def evaluate_case(predicted, expected):
    metrics = {}
    for field, target in expected.items():
        actual = predicted[field]
        exact = (actual is None and target is None) or (actual is not None and target is not None and actual['value'] == target['value'])
        metrics[field] = {'exact_match': bool(exact),
                          'bbox_iou': round(intersection_over_union(tuple(actual['bbox']),tuple(target['bbox'])),4) if actual and target else None,
                          'expected': target['value'] if target else None, 'predicted': actual['value'] if actual else None}
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT/'samples/ground_truth.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'samples/output')
    parser.add_argument('--model', default='Kapilydv6/layoutlmv3-invoice-parser')
    parser.add_argument('--adapter',default=str(ROOT/'assets/adaptation'))
    parser.add_argument('--tesseract-cmd')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--mode', choices=['hybrid','model','heuristic'], default='hybrid')
    parser.add_argument('--local-files-only',action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    pipeline = Pipeline(mode=args.mode,checkpoint=args.model,device=args.device,
                        tesseract_cmd=args.tesseract_cmd,local_files_only=args.local_files_only,adapter=args.adapter)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cases = []
    for case in manifest:
        path = args.manifest.parent/'input'/case['file']
        result,image,trace = pipeline.run(path,rotate=case.get('rotate',0))
        stem = path.stem
        image.save(args.output_dir/f'{stem}.png')
        for suffix,content in [('.json',result),('.trace.json',trace)]:
            (args.output_dir/(stem+suffix)).write_text(json.dumps(content,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        metrics = evaluate_case(result['fields'],case['fields'])
        exact = sum(field['exact_match'] for field in metrics.values())
        print(f'{stem}: exact {exact}/5; {result["meta"]["processing_time_sec"]:.2f}s; model_windows={trace["model_windows"]}',flush=True)
        cases.append({'file':case['file'],'split':case['split'],'metrics':metrics,'processing_time_sec':result['meta']['processing_time_sec']})
    scores = [metric for case in cases for metric in case['metrics'].values()]
    present = [metric for metric in scores if metric['expected'] is not None]
    absent = [metric for metric in scores if metric['expected'] is None]
    ious = [metric['bbox_iou'] if metric['bbox_iou'] is not None else 0 for metric in present]
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    dirty = bool(subprocess.run(['git','diff','--name-only'],cwd=ROOT,capture_output=True,text=True).stdout.strip())
    report = {'mode':args.mode,'model':pipeline.model_name,'git_commit':revision,'working_tree_modified':dirty,'python':platform.python_version(),
              'platform':platform.platform(),'invoice_count':len(cases),'field_count':len(scores),
              'exact_matches':sum(m['exact_match'] for m in scores),
              'present_field_exact_matches':sum(m['exact_match'] for m in present),'present_field_count':len(present),
              'absent_field_correct_nulls':sum(m['exact_match'] for m in absent),'absent_field_count':len(absent),
              'mean_iou_present_fields_misses_count_zero':round(sum(ious)/len(ious),4) if ious else None,
              'cases':cases}
    (args.output_dir/'evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__ == '__main__':
    main()
