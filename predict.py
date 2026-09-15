"""Assignment CLI: python predict.py --input invoice.jpg --output result.jpg --json result.json."""
import argparse
import json
import sys
import time
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description='Extract five invoice fields with OCR, LayoutLMv3 and merged boxes.')
    command.add_argument('--input', type=Path, required=True, help='JPG/PNG image or PDF (page 1 only)')
    command.add_argument('--output', type=Path, default=Path('result.jpg'), help='Annotated JPG/PNG with a side legend')
    command.add_argument('--json', type=Path, default=Path('result.json'), dest='json_path')
    command.add_argument('--debug', type=Path, help='Optional complete OCR, predictions, candidates and decision trace')
    command.add_argument('--mode', choices=['hybrid', 'model', 'heuristic'], default='hybrid',
                         help='hybrid runs the model plus documented rules; heuristic explicitly disables the model')
    command.add_argument('--model', default='Kapilydv6/layoutlmv3-invoice-parser', dest='checkpoint')
    command.add_argument('--revision', help='HF commit hash for a custom checkpoint; default model revision is pinned')
    command.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    command.add_argument('--tesseract-cmd', help='OCR executable path if tesseract is not on PATH')
    command.add_argument('--language', default='eng', help='Installed Tesseract language; model is English')
    command.add_argument('--psm', type=int, choices=[3, 6, 11, 12], default=3, help='Tesseract page segmentation mode')
    command.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0,
                         help='Clockwise correction before OCR; output boxes are mapped back to original pixels')
    command.add_argument('--pdf-dpi', type=int, default=150, help='PDF page 1 rasterization resolution')
    command.add_argument('--cache-dir', help='Optional Hugging Face model cache directory')
    command.add_argument('--local-files-only', action='store_true', help='Require already-downloaded model files')
    return command


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not args.input.is_file():
        print(f'Input file does not exist: {args.input}', file=sys.stderr)
        return 2
    destinations = [args.output.resolve(), args.json_path.resolve()]
    if args.debug:
        destinations.append(args.debug.resolve())
    if args.input.resolve() in destinations or len(destinations) != len(set(destinations)):
        print('Input, annotated output, JSON and debug paths must be distinct.', file=sys.stderr)
        return 2
    if args.pdf_dpi < 72 or args.pdf_dpi > 600:
        print('--pdf-dpi must be between 72 and 600.', file=sys.stderr)
        return 2
    if args.output.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        print('--output must have a .jpg, .jpeg or .png extension.', file=sys.stderr)
        return 2
    try:
        from src.pipeline import Pipeline
        started = time.perf_counter()
        print(f'Initializing mode={args.mode}; first model download may take a few minutes.', file=sys.stderr)
        pipeline = Pipeline(mode=args.mode, checkpoint=args.checkpoint, revision=args.revision, device=args.device,
                            tesseract_cmd=args.tesseract_cmd, language=args.language, psm=args.psm,
                            cache_dir=args.cache_dir, local_files_only=args.local_files_only)
        output, image, trace = pipeline.run(args.input, rotate=args.rotate, pdf_dpi=args.pdf_dpi)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output, **({'quality': 95} if args.output.suffix.lower() != '.png' else {}))
        # CLI wall time includes model initialization/download and image serialization.
        output['meta']['processing_time_sec'] = round(time.perf_counter()-started, 3)
        write_json(args.json_path, output)
        if args.debug:
            write_json(args.debug, trace)
        found = sum(value is not None for value in output['fields'].values())
        print(f'Extracted {found}/5 fields. JSON: {args.json_path}; image: {args.output}', file=sys.stderr)
        if args.mode == 'heuristic':
            print('LayoutLMv3 was explicitly disabled for this diagnostic run.', file=sys.stderr)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f'Extraction failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

