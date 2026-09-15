"""Real LayoutLMv3 inference; no random classification head or silent fallback."""
from pathlib import Path
import hashlib
import json

from .merge import normalize_box
from .types import LABEL_MAP, Prediction, Word

DEFAULT_MODEL = 'Kapilydv6/layoutlmv3-invoice-parser'
DEFAULT_REVISION = 'e464cfcadf6b766e105119c46e06f81ff1e4836a'
DEFAULT_ADAPTER = str(Path(__file__).resolve().parents[1] / 'assets' / 'adaptation')


class InvoiceModel:
    def __init__(self, checkpoint: str = DEFAULT_MODEL, revision: str | None = None,
                 device: str = 'auto', cache_dir: str | None = None, local_files_only: bool = False,
                 adapter: str | None = DEFAULT_ADAPTER):
        import torch
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; use --device cpu')
        self.device = device
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.revision = revision or (DEFAULT_REVISION if checkpoint == DEFAULT_MODEL else None)
        options = {'local_files_only': local_files_only, 'cache_dir': cache_dir}
        if not Path(checkpoint).is_dir() and self.revision:
            options['revision'] = self.revision
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            checkpoint, use_safetensors=True, **options,
        ).to(device).eval()
        self.adapter_name = None
        if adapter and adapter != 'none':
            from safetensors.torch import load_file
            directory = Path(adapter)
            metadata = json.loads((directory/'config.json').read_text(encoding='utf-8'))
            if Path(checkpoint).is_dir():
                with (Path(checkpoint)/'model.safetensors').open('rb') as stream:
                    sha = hashlib.file_digest(stream,'sha256').hexdigest()
                if sha != metadata['base_weights_sha256']:
                    raise ValueError('Adaptation requires its exact base weights; use --adapter none for another model.')
                self.revision = metadata['base_revision']
            elif checkpoint != metadata['base_model'] or self.revision != metadata['base_revision']:
                raise ValueError('Adaptation requires its pinned base checkpoint; use --adapter none for another model.')
            labels = metadata['labels']
            self.model.classifier = torch.nn.Linear(self.model.config.hidden_size,len(labels)).to(device)
            self.model.num_labels = len(labels)
            self.model.config.num_labels = len(labels)
            self.model.config.id2label = dict(enumerate(labels))
            self.model.config.label2id = {label:index for index,label in enumerate(labels)}
            state = load_file(str(directory/'weights.safetensors'))
            if set(state) != set(metadata['trained_tensor_names']):
                raise ValueError('Adaptation tensor manifest does not match weights')
            incompatible = self.model.load_state_dict(state,strict=False)
            if incompatible.unexpected_keys:
                raise ValueError('Unexpected tensors in adaptation')
            self.model.eval()
            self.adapter_name = 'invoice-adaptation-flat6'
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(checkpoint, **options)
        # This checkpoint has processor_config.json but no preprocessor_config.json.
        # Use the base model's documented RGB / 224 px / mean=std=0.5 defaults.
        self.image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        self.id2label = {int(key): value for key, value in self.model.config.id2label.items()}
        supported = {LABEL_MAP.get(label.removeprefix('B-').removeprefix('I-')) for label in self.id2label.values()}
        if not set(LABEL_MAP.values()).issubset(supported):
            raise ValueError('Checkpoint must provide labels for all five invoice fields; '
                             'a base model with a random head is not an invoice classifier.')
        self.checkpoint = checkpoint
        self.last_window_count = 0

    def predict(self, image, words: list[Word]) -> dict[int, Prediction]:
        import torch
        if not words:
            self.last_window_count = 0
            return {}
        encoding = self.tokenizer(
            [word.text for word in words], boxes=[normalize_box(word.bbox, image.size) for word in words],
            truncation=True, padding='max_length', max_length=512, stride=96,
            return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors='pt',
        )
        pixels = self.image_processor(images=image, return_tensors='pt')['pixel_values'].to(self.device)
        totals: dict[int, list] = {}
        self.last_window_count = len(encoding['input_ids'])
        with torch.inference_mode():
            for window in range(self.last_window_count):
                inputs = {name: encoding[name][window:window+1].to(self.device)
                          for name in ('input_ids', 'attention_mask', 'bbox')}
                probabilities = self.model(**inputs, pixel_values=pixels).logits[0].softmax(-1).cpu()
                word_ids = encoding.word_ids(batch_index=window)
                seen = set()
                for token, word_id in enumerate(word_ids):
                    # The default HF token-classification training supervises first subtokens.
                    # Ignore continuations and window fragments starting inside a word.
                    if word_id is None or word_id in seen or encoding['offset_mapping'][window, token, 0] != 0:
                        continue
                    seen.add(word_id)
                    totals.setdefault(word_id, []).append(probabilities[token])
        predictions = {}
        for index, word in enumerate(words):
            if index not in totals:
                raise RuntimeError(f'Tokenizer did not produce a first subtoken for OCR word {index}')
            probability = torch.stack(totals[index]).mean(0)
            winner = int(probability.argmax())
            scores: dict[str, float] = {}
            for label_id, label in self.id2label.items():
                base = label.removeprefix('B-').removeprefix('I-')
                key = LABEL_MAP.get(base, base)
                scores[key] = scores.get(key, 0.0) + float(probability[label_id])
            predictions[word.id] = Prediction(self.id2label[winner], float(probability[winner]), scores)
        return predictions
