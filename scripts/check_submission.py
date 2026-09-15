"""Validate the deliverable files, output schema and original-pixel box bounds."""
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.merge import union_box


def main():
    required = ['README.md','ERROR_ANALYSIS.md','VALIDATION.md','requirements.txt','predict.py',
                'src/ocr.py','src/model.py','src/aggregate.py','src/merge.py','src/visualize.py',
                'assets/adaptation/config.json','assets/adaptation/weights.safetensors']
    for name in required:
        if not (ROOT/name).is_file():
            raise AssertionError(f'Missing deliverable: {name}')
    schema = json.loads((ROOT/'schema.json').read_text(encoding='utf-8'))
    validator = jsonschema.Draft202012Validator(schema,format_checker=jsonschema.FormatChecker())
    validated = 0
    for path in sorted((ROOT/'samples').rglob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data,dict) or set(data) != {'meta','fields'}:
            continue
        validator.validate(data)
        width,height = data['meta']['image_size']
        for field,result in data['fields'].items():
            if result is None:
                continue
            for box in [result['bbox'],*result.get('line_boxes',[])]:
                x1,y1,x2,y2 = box
                assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, (path,field,box)
            if 'line_boxes' in result:
                assert list(union_box(tuple(box) for box in result['line_boxes'])) == result['bbox'], (path,field)
        assert path.with_suffix('.png').is_file() or path.with_suffix('.jpg').is_file(), path
        validated += 1
    assert len(list((ROOT/'samples/input').glob('*.png'))) >= 3
    for path in (ROOT/'assets').rglob('*'):
        if path.is_file():
            assert path.stat().st_size <= 100_000_000, f'Asset exceeds 100 MB: {path}'
    print(f'Validated {validated} result JSON files, all field boxes, and required deliverables.')


if __name__ == '__main__':
    main()
