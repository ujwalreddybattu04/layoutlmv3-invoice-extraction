import time
from dataclasses import asdict
from pathlib import Path

from .aggregate import aggregate
from .layout import readable_order
from .merge import restore_box
from .model import DEFAULT_MODEL, DEFAULT_REVISION, DEFAULT_ADAPTER, InvoiceModel
from .ocr import TesseractOCR, load_image
from .types import FIELDS
from .visualize import annotate


class Pipeline:
    """Reuse OCR/model initialization when evaluating a collection of invoices."""
    def __init__(self, *, mode='hybrid', checkpoint=DEFAULT_MODEL, revision=None, device='auto',
                 tesseract_cmd=None, language='eng', psm=3, cache_dir=None, local_files_only=False,
                 adapter=DEFAULT_ADAPTER):
        self.mode = mode
        self.ocr = TesseractOCR(tesseract_cmd, language, psm)
        self.model = None if mode == 'heuristic' else InvoiceModel(
            checkpoint, revision, device, cache_dir, local_files_only, adapter)
        self.model_name = (f'{checkpoint}@{self.model.revision}' if self.model and self.model.revision
                           else checkpoint if self.model else 'none (explicit heuristic-only mode)')
        if self.model and self.model.adapter_name:
            self.model_name = f'{DEFAULT_MODEL}@{self.model.revision}+{self.model.adapter_name}'

    def run(self, input_path: Path, *, rotate: int = 0, pdf_dpi: int = 150):
        started = time.perf_counter()
        original = load_image(input_path, pdf_dpi)
        working = original.rotate(-rotate, expand=True) if rotate else original
        words = readable_order(self.ocr(working))
        predictions = self.model.predict(working, words) if self.model else {}
        fields, trace = aggregate(words, predictions, use_heuristics=self.mode != 'model')
        if rotate:
            for value in fields.values():
                if value:
                    value['bbox'] = list(restore_box(tuple(value['bbox']), original.size, rotate))
                    if 'line_boxes' in value:
                        value['line_boxes'] = [list(restore_box(tuple(box), original.size, rotate)) for box in value['line_boxes']]
        trace.update({'input_file': str(input_path), 'mode': self.mode, 'rotation_clockwise': rotate,
                      'working_image_size': list(working.size), 'trace_box_space': 'rotated working-image pixels',
                      'model': self.model_name, 'model_executed': bool(self.model and words),
                      'model_windows': self.model.last_window_count if self.model else 0,
                      'words': [asdict(word) for word in words],
                      'predictions': {str(key): asdict(value) for key, value in predictions.items()}})
        image = annotate(original, fields, mode=self.mode)
        output = {'meta': {'input_file': input_path.name, 'image_size': list(original.size),
                           'ocr_engine': f'tesseract-{self.ocr.version}', 'model': self.model_name,
                           'processing_time_sec': round(time.perf_counter()-started, 3)}, 'fields': fields}
        return output, image, trace
