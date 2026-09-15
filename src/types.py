from dataclasses import dataclass, field

Box = tuple[int, int, int, int]
FIELDS = ('vendor_name', 'company_address', 'invoice_number', 'invoice_date', 'total')
LABEL_MAP = {
    'VENDOR_NAME': 'vendor_name', 'VENDOR_ADDR': 'company_address',
    'COMPANY_ADDRESS': 'company_address', 'INVOICE_NUM': 'invoice_number',
    'INVOICE_NUMBER': 'invoice_number', 'INVOICE_DATE': 'invoice_date', 'TOTAL': 'total',
}


@dataclass(frozen=True)
class Word:
    id: int
    text: str
    bbox: Box
    confidence: float
    line: str = ''
    # One character box per Python character, when Tesseract supplies alignment.
    char_boxes: tuple[Box, ...] = ()

    @property
    def height(self) -> int:
        return max(1, self.bbox[3] - self.bbox[1])


@dataclass(frozen=True)
class Prediction:
    label: str = 'O'
    confidence: float = 1.0
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def field(self) -> str | None:
        return LABEL_MAP.get(self.label.removeprefix('B-').removeprefix('I-'))


@dataclass(frozen=True)
class Piece:
    word_id: int
    text: str
    bbox: Box
    ocr_confidence: float
    start: int = 0
    end: int = 0
    approximate_box: bool = False


@dataclass
class Candidate:
    field: str
    pieces: list[Piece]
    source: str
    anchor: str = ''
    anchor_strength: float = 0.0
    rank: float = 0.0
    reasons: list[str] = field(default_factory=list)

