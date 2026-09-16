"""Reproduce invoice and robustness checks; use on an extracted submission."""
from pathlib import Path
import copy, hashlib, json, os, subprocess, sys, time
from PIL import Image, ImageDraw
import fitz, jsonschema

import argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/final-audit')
parser.add_argument('--device',default='cpu',choices=['cpu','cuda','auto'])
parser.add_argument('--model',default='Kapilydv6/layoutlmv3-invoice-parser')
parser.add_argument('--tesseract-cmd')
parser.add_argument('--local-files-only',action='store_true')
args=parser.parse_args()
OUT=args.output_dir.resolve()
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from src.pipeline import Pipeline
from src.merge import union_box,restore_box
from src.types import Word

schema=json.loads((ROOT/'schema.json').read_text())
validator=jsonschema.Draft202012Validator(schema,format_checker=jsonschema.FormatChecker())
checks=[]; cases=[]
def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def valid(data):
    validator.validate(data)
    w,h=data['meta']['image_size']
    for value in data['fields'].values():
        if value:
            for b in [value['bbox'],*value.get('line_boxes',[])]:
                x1,y1,x2,y2=b
                assert 0<=x1<x2<=w and 0<=y1<y2<=h,b
            if 'line_boxes' in value:
                assert list(union_box(value['line_boxes']))==value['bbox']

print('Loading real LayoutLMv3 + bundled adaptation...',flush=True)
pipeline=Pipeline(device=args.device,checkpoint=args.model,tesseract_cmd=args.tesseract_cmd,local_files_only=args.local_files_only)
def run(name,path,expected,group,rotate=0):
    data,image,trace=pipeline.run(path,rotate=rotate)
    valid(data)
    image.save(OUT/(name+'.png'))
    save(name+'.json',data);save(name+'.trace.json',trace)
    matches={k:((data['fields'][k]['value'] if data['fields'][k] else None)==(v['value'] if v else None)) for k,v in expected.items()}
    entry={'name':name,'group':group,'exact':sum(matches.values()),'fields':len(matches),'seconds':data['meta']['processing_time_sec'],
        'mismatches':{k:{'expected':expected[k]['value'] if expected[k] else None,'actual':data['fields'][k]['value'] if data['fields'][k] else None} for k,ok in matches.items() if not ok},
        'schema_and_box_bounds':'pass','model_executed':trace['model_executed']}
    cases.append(entry);print(json.dumps(entry,ensure_ascii=False),flush=True)
    save('progress.json',cases)
    return data,trace

for group,manifest in [('development','samples/ground_truth.json'),('reviewed_layouts','samples/heldout/ground_truth.json'),('stress','samples/stress/ground_truth.json')]:
    mpath=ROOT/manifest
    for item in json.loads(mpath.read_text()):
        result,trace=run(group+'_'+Path(item['file']).stem,mpath.parent/'input'/item['file'],item['fields'],group)
        if group=='development' and item['file']=='classic.png':
            baseline=result; truth=item['fields']

source=ROOT/'samples/input/classic.png'
original=Image.open(source).convert('RGB')
variants=OUT/'inputs';variants.mkdir(exist_ok=True)
for name,image,fmt in [
    ('jpeg_quality_65',original,'JPEG'),
    ('grayscale',original.convert('L'),'PNG'),
    ('half_resolution',original.resize((600,800),Image.Resampling.LANCZOS),'PNG'),
    ('rgba',original.convert('RGBA'),'PNG'),
]:
    path=variants/(name+('.jpg' if fmt=='JPEG' else '.png'))
    image.save(path,format=fmt,**({'quality':65} if fmt=='JPEG' else {}))
    run(name,path,truth,'new_variants')

for angle in (90,180,270):
    path=variants/f'rotation_{angle}.png'
    original.rotate(angle,expand=True).save(path)
    result,_=run(f'rotation_{angle}',path,truth,'new_variants',rotate=angle)
    assert result['fields'].keys()==baseline['fields'].keys()
    for k,v in baseline['fields'].items():
        if v:
            assert result['fields'][k]['value']==v['value']
            assert result['fields'][k]['bbox']==list(restore_box(tuple(v['bbox']),Image.open(path).size,angle))
    checks.append(f'{angle}-degree correction: exact values and original-pixel box mapping')

for field in ('invoice_number','invoice_date'):
    im=original.copy();draw=ImageDraw.Draw(im);b=baseline['fields'][field]['bbox']
    draw.rectangle((b[0]-2,b[1]-2,b[2]+2,b[3]+2),fill='white')
    path=variants/f'missing_{field}.png';im.save(path)
    expected=copy.deepcopy(truth);expected[field]=None
    run('missing_'+field,path,expected,'new_variants')

pdf=fitz.open()
for src in (source,ROOT/'samples/input/ledger.png'):
    im=Image.open(src);pg=pdf.new_page(width=im.width*72/150,height=im.height*72/150)
    pg.insert_image(pg.rect,filename=str(src))
pdfpath=variants/'two_page_invoice.pdf';pdf.save(pdfpath);pdf.close()
data,_=run('pdf_first_page',pdfpath,truth,'new_variants')
checks.append('Two-page PDF returns first-page invoice, not second-page content')

words=[Word(i,'sample',(20+(i%15)*75,20+(i//15)*25,85+(i%15)*75,40+(i//15)*25),.99) for i in range(750)]
predictions=pipeline.model.predict(Image.new('RGB',(1200,1600),'white'),words)
assert set(predictions)==set(range(750)) and pipeline.model.last_window_count>1
checks.append(f'750 words: {pipeline.model.last_window_count} windows; every word predicted')

# Full separate-process CLI using the exact assignment entry point and flags.
command=[sys.executable,'predict.py','--input','samples/input/classic.png','--output',str(OUT/'cli.jpg'),'--json',str(OUT/'cli.json')]
command += ['--device',args.device]
if args.model != 'Kapilydv6/layoutlmv3-invoice-parser': command += ['--model',args.model]
if args.tesseract_cmd: command += ['--tesseract-cmd',args.tesseract_cmd]
if args.local_files_only: command += ['--local-files-only']
print('Running separate-process assignment CLI...',flush=True)
started=time.perf_counter(); proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=300)
assert proc.returncode==0,proc.stderr
cli=json.loads((OUT/'cli.json').read_text()); valid(cli)
assert cli['fields']==baseline['fields']
checks.append('Exact assignment CLI: exit 0; JSON fields match pipeline; annotated JPG exists')
save('cli_process.json',{'command':command,'returncode':proc.returncode,'stderr':proc.stderr,'wall_seconds':round(time.perf_counter()-started,2)})

bad=variants/'corrupt.png';bad.write_bytes(b'not an image')
bad_cases=[('missing_input',['--input',str(variants/'absent.png')],2),
 ('input_overwrite',['--input',str(source),'--output',str(source)],2),
 ('output_collision',['--input',str(source),'--output',str(OUT/'same.png'),'--json',str(OUT/'same.png')],2),
 ('bad_pdf_dpi',['--input',str(source),'--pdf-dpi','10'],2),
 ('bad_output_type',['--input',str(source),'--output',str(OUT/'result.gif')],2),
 ('corrupt_input',['--input',str(bad),'--mode','heuristic'],1)]
before=hashlib.sha256(source.read_bytes()).hexdigest()
for name,bad_args,expected_code in bad_cases:
    proc=subprocess.run([sys.executable,'predict.py',*bad_args]+(['--tesseract-cmd',args.tesseract_cmd] if args.tesseract_cmd else []),cwd=ROOT,capture_output=True,text=True,timeout=90)
    assert proc.returncode==expected_code,(name,proc.returncode,proc.stderr)
    assert 'Traceback' not in proc.stderr,(name,proc.stderr)
    checks.append(name+': clear error, expected exit code, no traceback')
assert hashlib.sha256(source.read_bytes()).hexdigest()==before

source_files=sorted([ROOT/'predict.py',*(ROOT/'src').glob('*.py')])
source_hash=hashlib.sha256(b''.join(p.relative_to(ROOT).as_posix().encode()+b'\0'+p.read_bytes().replace(b'\r\n',b'\n') for p in source_files)).hexdigest()
report={'extraction_source_sha256':source_hash,
 'python':sys.version,'device':args.device,'model':pipeline.model_name,'case_count':len(cases),
 'cases':cases,'runtime_checks':checks,'scope':'Synthetic fixtures and transformed variants; reviewed layouts are not unseen evaluation.'}
save('audit_report.json',report)
print('AUDIT COMPLETE',flush=True)
print(json.dumps({'cases':len(cases),'exact':sum(x['exact'] for x in cases),'fields':sum(x['fields'] for x in cases),'runtime_checks':len(checks)},indent=2),flush=True)
