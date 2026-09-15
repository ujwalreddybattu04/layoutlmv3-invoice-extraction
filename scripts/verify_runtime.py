"""Additional real-runtime checks for blank pages, long token streams and rotation."""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.model import DEFAULT_MODEL
from src.pipeline import Pipeline
from src.types import FIELDS, Word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default=DEFAULT_MODEL)
    parser.add_argument('--tesseract-cmd')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/runtime-checks')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pipeline = Pipeline(checkpoint=args.model,tesseract_cmd=args.tesseract_cmd,device=args.device)
    checks = {}
    blank = args.output_dir/'blank.png'
    Image.new('RGB',(800,1000),'white').save(blank)
    result,_,trace = pipeline.run(blank)
    assert result['fields'] == dict.fromkeys(FIELDS)
    assert trace['model_executed'] is False
    checks['blank_page_five_nulls'] = True

    image = Image.new('RGB',(1200,1600),'white')
    words = [Word(i,'sample',(20+(i%15)*75,20+(i//15)*25,85+(i%15)*75,40+(i//15)*25),0.99)
             for i in range(750)]
    predictions = pipeline.model.predict(image,words)
    assert set(predictions) == {word.id for word in words}
    assert pipeline.model.last_window_count > 1
    checks['long_sequence'] = {'ocr_words':len(words),'predicted_words':len(predictions),
                                'windows':pipeline.model.last_window_count,'last_word_preserved':749 in predictions}

    original = ROOT/'samples/input/classic.png'
    baseline,_,_ = pipeline.run(original)
    rotated = args.output_dir/'classic_sideways.png'
    with Image.open(original) as source:
        source.rotate(90,expand=True).save(rotated)
    corrected,annotated,_ = pipeline.run(rotated,rotate=90)
    assert {k:v['value'] if v else None for k,v in corrected['fields'].items()} == {
        k:v['value'] if v else None for k,v in baseline['fields'].items()}
    for field,value in baseline['fields'].items():
        if value:
            x1,y1,x2,y2 = value['bbox']
            assert corrected['fields'][field]['bbox'] == [y1,1200-x2,y2,1200-x1]
    annotated.save(args.output_dir/'classic_sideways_annotated.png')
    checks['rotation_values_and_original_pixel_boxes'] = True
    checks['model'] = pipeline.model_name
    path = args.output_dir/'runtime_checks.json'
    path.write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
