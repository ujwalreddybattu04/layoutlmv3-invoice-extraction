import json
import subprocess
import sys

import pytest
from PIL import Image

from src.merge import normalize_box, denormalize_box, restore_box, union_box
from src.ocr import HOCRParser, load_image


def test_normalization_and_inverse_bound_rounding_error():
    size = (1240,1754)
    box = (123,456,789,1200)
    normalized = normalize_box(box,size)
    assert all(0 <= value <= 1000 for value in normalized)
    restored = denormalize_box(tuple(normalized),size)
    assert all(abs(a-b) <= 2 for a,b in zip(box,restored))


@pytest.mark.parametrize('angle,box', [
    (0,(10,20,40,60)), (90,(140,10,180,40)), (180,(60,140,90,180)), (270,(20,60,60,90)),
])
def test_inverse_rotation(angle,box):
    assert restore_box(box,(100,200),angle) == (10,20,40,60)


def test_empty_union_is_a_programming_error():
    with pytest.raises(ValueError):
        union_box([])


def test_hocr_unicode_character_alignment():
    html = '''<html><body><span class="ocr_line" id="line_1"><span class="ocrx_word" title="bbox 10 20 30 40; x_wconf 91"><span class="ocrx_cinfo" title="x_bboxes 10 20 20 40">₹</span><span class="ocrx_cinfo" title="x_bboxes 20 20 30 40">5</span></span></span></body></html>'''
    parser = HOCRParser((100,100))
    parser.feed(html)
    assert parser.words[0].text == '₹5'
    assert parser.words[0].confidence == 0.91
    assert len(parser.words[0].char_boxes) == 2


def test_transparency_is_composited_on_white(tmp_path):
    path = tmp_path/'transparent.png'
    Image.new('RGBA',(10,10),(0,0,0,0)).save(path)
    assert load_image(path).getpixel((0,0)) == (255,255,255)


def test_cli_refuses_overwriting_input(tmp_path):
    path = tmp_path/'input.png'
    Image.new('RGB',(10,10),'white').save(path)
    before = path.read_bytes()
    result = subprocess.run([sys.executable,'predict.py','--input',str(path),'--output',str(path)],capture_output=True)
    assert result.returncode == 2
    assert path.read_bytes() == before


def test_cli_reports_missing_input_without_loading_model(tmp_path):
    result = subprocess.run([sys.executable,'predict.py','--input',str(tmp_path/'missing.png')],capture_output=True,text=True)
    assert result.returncode == 2
    assert 'does not exist' in result.stderr

