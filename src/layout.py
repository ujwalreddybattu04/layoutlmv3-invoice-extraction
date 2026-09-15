"""Small geometry and text helpers shared by candidate generation and output."""
import re
from dataclasses import dataclass
from statistics import median

from .merge import union_box, vertical_overlap
from .types import Piece, Word


@dataclass
class Row:
    words: list[Word]

    @property
    def bbox(self):
        return union_box(word.bbox for word in self.words)

    @property
    def height(self):
        return median(word.height for word in self.words)

    @property
    def text(self):
        return ' '.join(word.text for word in self.words)

    def offsets(self):
        offset = 0
        for word in self.words:
            yield word, offset, offset + len(word.text)
            offset += len(word.text) + 1

    def pieces(self, start: int = 0, end: int | None = None) -> list[Piece]:
        end = len(self.text) if end is None else end
        return [slice_word(word, max(0, start-left), min(len(word.text), end-left))
                for word, left, right in self.offsets() if left < end and right > start]


def slice_word(word: Word, start: int = 0, end: int | None = None) -> Piece:
    end = len(word.text) if end is None else end
    approximate = False
    box = word.bbox
    if start > 0 or end < len(word.text):
        if len(word.char_boxes) == len(word.text):
            box = union_box(word.char_boxes[start:end])
        else:
            # Retain the full measured word box. Proportional character widths would
            # imply precision OCR never supplied; expose this limitation in the trace.
            approximate = True
    return Piece(word.id, word.text[start:end], box, word.confidence, start, end, approximate)


def make_rows(words: list[Word]) -> list[Row]:
    rows: list[Row] = []
    for word in sorted(words, key=lambda w: ((w.bbox[1]+w.bbox[3])/2, w.bbox[0])):
        candidates = [row for row in rows[-8:] if vertical_overlap(word.bbox, row.bbox) >= 0.5
                      and abs((word.bbox[1]+word.bbox[3])-(row.bbox[1]+row.bbox[3]))/2 <= max(word.height, row.height)*0.6]
        if candidates:
            min(candidates, key=lambda row: abs(row.bbox[1]-word.bbox[1])).words.append(word)
        else:
            rows.append(Row([word]))
    for row in rows:
        row.words.sort(key=lambda word: word.bbox[0])
    return sorted(rows, key=lambda row: (row.bbox[1], row.bbox[0]))


def segments(row: Row) -> list[Row]:
    result: list[Row] = []
    for word in row.words:
        if not result or word.bbox[0] - result[-1].words[-1].bbox[2] > 3 * row.height:
            result.append(Row([word]))
        else:
            result[-1].words.append(word)
    return result


def piece_lines(pieces: list[Piece]) -> list[list[Piece]]:
    rows = make_rows([Word(i, p.text, p.bbox, p.ocr_confidence) for i, p in enumerate(pieces)])
    return [[pieces[word.id] for word in row.words] for row in rows]


def stitch(pieces: list[Piece], field: str) -> str:
    """Preserve OCR characters; recover separators from syntax and observed gaps."""
    lines = []
    for line in piece_lines(pieces):
        text = ''
        previous = None
        for piece in line:
            separator = ' ' if text else ''
            if previous:
                h = max(1, min(piece.bbox[3]-piece.bbox[1], previous.bbox[3]-previous.bbox[1]))
                gap = piece.bbox[0] - previous.bbox[2]
                if piece.word_id == previous.word_id and piece.start == previous.end:
                    separator = ''
                elif field in ('invoice_number', 'invoice_date', 'total'):
                    if (piece.text[:1] in '-/.,:' or previous.text[-1:] in '-/.,:' or
                            previous.text in ('$', '€', '£', '₹') or gap <= 0.16*h):
                        separator = ''
                elif piece.text[:1] in ',.;:)' or previous.text[-1:] == '(':
                    separator = ''
            text += separator + piece.text
            previous = piece
        lines.append(text)
    return '\n'.join(lines)


def readable_order(words: list[Word]) -> list[Word]:
    return [word for row in make_rows(words) for word in row.words]

