"""Tesseract hOCR preserves real word boxes and optional character boxes."""
import os
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

from .merge import clip_box
from .types import Word


class HOCRParser(HTMLParser):
    def __init__(self, size: tuple[int, int]):
        super().__init__(convert_charrefs=True)
        self.size = size
        self.words: list[Word] = []
        self.stack: list[dict] = []
        self.line = ''
        self.current: dict | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        entry = {'tag': tag, 'attrs': attributes}
        self.stack.append(entry)
        classes = attributes.get('class', '').split()
        if 'ocr_line' in classes or 'ocr_header' in classes:
            self.line = attributes.get('id', '')
        if 'ocrx_word' in classes:
            self.current = {'depth': len(self.stack), 'attrs': attributes, 'text': [], 'boxes': []}

    def handle_data(self, data):
        if self.current is None or not data.strip():
            return
        self.current['text'].append(data)
        title = self.stack[-1]['attrs'].get('title', '')
        match = re.search(r'x_bboxes\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title)
        if match:
            self.current['boxes'].extend([clip_box(map(int, match.groups()), self.size)] * len(data))

    def handle_endtag(self, tag):
        if self.current is not None and len(self.stack) == self.current['depth']:
            text = ''.join(self.current['text']).strip()
            title = self.current['attrs'].get('title', '')
            match = re.search(r'\bbbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title)
            confidence = re.search(r'x_wconf\s+([\d.]+)', title)
            if text and match:
                box = clip_box(map(int, match.groups()), self.size)
                char_boxes = tuple(self.current['boxes'])
                if len(char_boxes) != len(text):
                    char_boxes = ()
                if box[2] > box[0] and box[3] > box[1]:
                    self.words.append(Word(len(self.words), text, box,
                                           min(1.0, float(confidence[1])/100) if confidence else 0.0,
                                           self.line, char_boxes))
            self.current = None
        if self.stack:
            self.stack.pop()

    def handle_startendtag(self, tag, attrs):
        pass


class TesseractOCR:
    def __init__(self, executable: str | None = None, language: str = 'eng', psm: int = 3):
        import pytesseract
        candidate = executable or os.environ.get('TESSERACT_CMD') or shutil.which('tesseract')
        if not candidate:
            raise RuntimeError('Tesseract was not found. Install Tesseract 5 and add it to PATH, '
                               'or pass --tesseract-cmd /path/to/tesseract. See README Setup.')
        pytesseract.pytesseract.tesseract_cmd = str(candidate)
        self.language, self.psm = language, psm
        self.version = str(pytesseract.get_tesseract_version()).splitlines()[0]

    def __call__(self, image: Image.Image) -> list[Word]:
        import pytesseract
        content = pytesseract.image_to_pdf_or_hocr(
            image, extension='hocr', lang=self.language,
            config=f'--psm {self.psm} -c hocr_char_boxes=1', timeout=90,
        )
        parser = HOCRParser(image.size)
        parser.feed(content.decode('utf-8'))
        return parser.words


def load_image(path: Path, pdf_dpi: int = 150) -> Image.Image:
    if path.suffix.lower() == '.pdf':
        import fitz
        with fitz.open(path) as document:
            if document.needs_pass:
                raise ValueError('Password-protected PDFs are unsupported')
            if not document.page_count:
                raise ValueError('The PDF has no pages')
            pixmap = document[0].get_pixmap(dpi=pdf_dpi, alpha=False, colorspace=fitz.csRGB)
            return Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples)
    with Image.open(path) as source:
        # Composite transparent PNGs against paper white, not black.
        if source.mode in ('RGBA', 'LA') or 'transparency' in source.info:
            rgba = source.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            return Image.alpha_composite(background, rgba).convert('RGB')
        return source.convert('RGB')

