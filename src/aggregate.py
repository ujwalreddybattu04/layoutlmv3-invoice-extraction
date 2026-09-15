"""Generate bounded candidates, rank evidence, then abstain or select one per field.

Heuristics operate only on OCR text. They never invent a missing value or silently
replace the model; the optional trace contains every candidate and its evidence.
"""
import re
from datetime import datetime
from statistics import mean

from .layout import Row, make_rows, piece_lines, segments, slice_word, stitch
from .merge import union_box, vertical_overlap
from .types import Candidate, FIELDS, Piece, Prediction, Word

# Longest matches win; negative keys mask positive substrings (e.g. Sub Total).
KEYS = [
    ('blocked_total', r'\b(?:sub\s*total|balance\s+due|amount\s+due|total\s+tax|tax\s+total|total\s+paid|paid|discount|(?:sales\s+)?tax|vat|[cis]gst)\b', 1.0),
    ('blocked_date', r'\b(?:due\s+date|delivery\s+date|dispatch\s+date|ship(?:ping)?\s+date|payment\s+due)\b', 1.0),
    ('buyer', r'\b(?:bill(?:ed)?\s+to|ship\s+to|sold\s+to|deliver\s+to|customer(?:\s+(?:name|address))?|buyer)\b', 1.0),
    ('invoice_number', r'\b(?:invoice\s*(?:number|no\.?|id|#)|inv\.?\s*(?:no\.?|#)|bill\s*(?:number|no\.?|#))', 1.0),
    ('invoice_date', r'\b(?:invoice\s+date|date\s+of\s+issue|issue\s+date|issued\s+on|dated)\b', 1.0),
    ('invoice_date', r'\bdate\b', 0.75),
    ('total', r'\b(?:grand\s+total|invoice\s+total|total\s+(?:amount(?:\s+payable)?|payable|incl\.?\s+tax)|amount\s+payable)\b', 1.0),
    ('total', r'\btotal\b', 0.8),
    ('company_address', r'\b(?:vendor|supplier|company|seller)\s+address\b', 1.0),
    ('vendor_name', r'\b(?:vendor(?:\s+name)?|supplier(?:\s+name)?|seller|issued\s+by|sold\s+by|from)\b', 0.95),
]
DATE = re.compile(r'(?<!\w)(?:\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}|\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s*,?\s*\d{4}|[A-Za-z]{3,9}\s+\d{1,2}\s*,?\s*\d{4})(?!\w)')
MONEY = re.compile(r'(?<![\w/])(?:(?:USD|INR|EUR|GBP|CAD|AUD|Rs\.?)\s*|[$€£₹]\s*)?\(?-?\d[\d,. ]*(?:\d|(?<=\d))\)?(?!\w)', re.I)
IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9_./-]*(?:\s*[-/]\s*[A-Za-z0-9_./-]+)*')
CONTACT = re.compile(r'\b(?:email|phone|tel|fax|www|https?|gstin|vat\s+id|tax\s+id|bank|iban|swift|account|registration)\b|@', re.I)
ADDRESS_HINT = re.compile(r'\b(?:road|rd\.?|street|st\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|park|floor|suite|building|blvd|boulevard|industrial|nagar|sector|way|pincode|postcode)\b', re.I)


def anchors_for(row: Row) -> list[dict]:
    hits = []
    for field, pattern, strength in KEYS:
        for match in re.finditer(pattern, row.text, re.I):
            hits.append({'field': field, 'start': match.start(), 'end': match.end(),
                         'text': match.group(), 'strength': strength,
                         'bbox': union_box(p.bbox for p in row.pieces(match.start(), match.end()))})
    chosen = []
    for hit in sorted(hits, key=lambda h: (-(h['end']-h['start']), h['start'])):
        if not any(hit['start'] < other['end'] and hit['end'] > other['start'] for other in chosen):
            chosen.append(hit)
    return sorted(chosen, key=lambda hit: hit['start'])


def weighted_score(pieces: list[Piece], predictions: dict[int, Prediction], field: str) -> float:
    total = sum(max(1, len(p.text)) for p in pieces)
    return sum(max(1, len(p.text)) * predictions.get(p.word_id, Prediction()).scores.get(field, 0.0)
               for p in pieces) / max(1, total)


def ocr_score(pieces: list[Piece]) -> float:
    total = sum(max(1, len(p.text)) for p in pieces)
    return sum(max(1, len(p.text))*p.ocr_confidence for p in pieces) / max(1, total)


def plausible_address(text: str) -> bool:
    return bool(ADDRESS_HINT.search(text) or (re.search(r'\d', text) and re.search(r'[A-Za-z]{3}', text)))


def normalize_date(value: str) -> str | None:
    value = re.sub(r'\s*([/.-])\s*', r'\1', value).replace(',', '')
    # Ambiguous numeric dates are deliberately left unnormalized.
    match = re.fullmatch(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', value)
    if match and int(match[1]) <= 12 and int(match[2]) <= 12 and match[1] != match[2]:
        return None
    for form in ('%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%d/%m/%Y', '%d-%m-%Y',
                 '%d.%m.%Y', '%m/%d/%Y', '%m-%d-%Y', '%d %B %Y', '%d %b %Y', '%B %d %Y', '%b %d %Y'):
        try:
            return datetime.strptime(value, form).date().isoformat()
        except ValueError:
            continue
    return None


def plausible(field: str, value: str) -> bool:
    if not value or len(value) > 500:
        return False
    if field == 'invoice_number':
        return bool(IDENTIFIER.fullmatch(value)) and 2 <= len(value) <= 64 and not DATE.fullmatch(value)
    if field == 'invoice_date':
        if not DATE.fullmatch(value):
            return False
        # Reject obviously invalid calendar dates, while allowing ambiguous DD/MM.
        numeric = re.fullmatch(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', re.sub(r'\s', '', value))
        if numeric and int(numeric[1]) <= 12 and int(numeric[2]) <= 12:
            return min(int(numeric[1]), int(numeric[2])) > 0
        return normalize_date(value) is not None
    if field == 'total':
        return bool(MONEY.fullmatch(value)) and len(value) <= 40 and not re.search(r'\d\s+\d{4,}', value)
    if field == 'vendor_name':
        return (bool(re.search(r'[A-Za-z]{2}', value)) and not CONTACT.search(value)
                and not plausible_address(value) and value.lower() not in
                {'for','from','to','thank','thanks','you','your','payment','terms','invoice','total','date'})
    return bool(re.search(r'[A-Za-z]{2}', value)) and not CONTACT.search(value) and plausible_address(value)


def select_value(row: Row, start: int, end: int, field: str) -> list[Piece]:
    text = row.text[start:end]
    leading = len(text) - len(text.lstrip(' \t:#=-'))
    start += leading
    text = row.text[start:end].rstrip()
    end = start + len(text)
    pattern = {'invoice_number': IDENTIFIER, 'invoice_date': DATE, 'total': MONEY}.get(field)
    if pattern:
        match = pattern.search(text)
        if not match:
            return []
        # An anchored number must immediately follow the key, not appear in unrelated prose.
        if text[:match.start()].strip(' :#=-'):
            return []
        start, end = start+match.start(), start+match.end()
    return row.pieces(start, end) if start < end else []


def same_column(first, second, height):
    return abs(first[0]-second[0]) <= 2.0*height


def address_after(seed: list[Piece], rows: list[Row]) -> list[Piece]:
    result = []
    previous = union_box(p.bbox for p in seed)
    start_x = previous[0]
    height = max(p.bbox[3]-p.bbox[1] for p in seed)
    for row in rows:
        if row.bbox[1] < previous[3] - 0.25*height:
            continue
        if row.bbox[1]-previous[3] > 2.0*height:
            break
        possible = [segment for segment in segments(row)
                    if abs(segment.bbox[0]-start_x) <= 2*height]
        if not possible:
            continue
        segment = min(possible, key=lambda s: abs(s.bbox[0]-start_x))
        if anchors_for(segment) or CONTACT.search(segment.text):
            break
        if not result and not plausible_address(segment.text):
            break
        if not re.search(r'[A-Za-z]', segment.text) and not re.fullmatch(r'[\d -]{4,12}', segment.text):
            break
        result.extend(segment.pieces())
        previous = segment.bbox
        height = row.height
        if len(piece_lines(result)) >= 4:
            break
    return result


def key_candidates(rows: list[Row]) -> list[Candidate]:
    candidates = []
    for row_index, row in enumerate(rows):
        anchors = anchors_for(row)
        for index, anchor in enumerate(anchors):
            field = anchor['field']
            if field not in FIELDS:
                continue
            end = anchors[index+1]['start'] if index+1 < len(anchors) else len(row.text)
            pieces = select_value(row, anchor['end'], end, field)
            if not pieces:
                # A label above a value: nearest aligned segment within two text heights.
                for following in rows[row_index+1:row_index+3]:
                    if following.bbox[1]-row.bbox[3] > 2.2*row.height:
                        break
                    options = [s for s in segments(following)
                               if (abs(s.bbox[0]-anchor['bbox'][0]) <= 2*row.height or
                                   (field in ('total', 'invoice_number', 'invoice_date') and
                                    anchor['bbox'][0] <= s.bbox[0] <= anchor['bbox'][2]+8*row.height))
                               and not anchors_for(s)]
                    if options:
                        nearest = min(options, key=lambda s: abs(s.bbox[0]-anchor['bbox'][0]))
                        pieces = select_value(nearest, 0, len(nearest.text), field)
                        break
            if pieces:
                candidate = Candidate(field, pieces, 'heuristic', anchor['text'], anchor['strength'],
                                      reasons=['OCR value linked to an explicit field key'])
                if field == 'company_address':
                    candidate.pieces += address_after(pieces, rows)
                candidates.append(candidate)
    return candidates


def model_candidates(rows: list[Row], predictions: dict[int, Prediction]) -> list[Candidate]:
    candidates = []
    for row in rows:
        for segment in segments(row):
            current: Candidate | None = None
            for index, word in enumerate(segment.words):
                prediction = predictions.get(word.id, Prediction())
                field = prediction.field
                strength = prediction.scores.get(field, 0.0) if field else 0.0
                following = predictions.get(segment.words[index+1].id, Prediction()) if index+1 < len(segment.words) else Prediction()
                if (current and field != current.field and following.field == current.field
                        and prediction.scores.get(current.field, 0) >= 0.15 and word.confidence >= 0.15):
                    current.pieces.append(slice_word(word))
                    current.reasons.append('Bridged one uncertain interior word')
                    continue
                if strength < 0.50 or word.confidence < 0.15:
                    # Bridge a single uncertain interior word only with model support.
                    if current and following.field == current.field and prediction.scores.get(current.field, 0) >= 0.15 and word.confidence >= 0.15:
                        current.pieces.append(slice_word(word))
                        current.reasons.append('Bridged one uncertain interior word')
                        continue
                    current = None
                    continue
                if current is None or current.field != field:
                    current = Candidate(field, [], 'model', reasons=['Adjacent model-labelled OCR words'])
                    candidates.append(current)
                elif prediction.label.startswith('B-'):
                    current.reasons.append('Repaired adjacent repeated B label within the same text segment')
                current.pieces.append(slice_word(word))
    # Offer multiline model spans only in the same column with a short vertical gap.
    for field in ('vendor_name', 'company_address'):
        groups = [c for c in candidates if c.field == field and plausible(field, stitch(c.pieces, field))]
        for first in groups:
            combined = list(first.pieces)
            previous = union_box(p.bbox for p in combined)
            for second in groups:
                box = union_box(p.bbox for p in second.pieces)
                height = max(1, previous[3]-previous[1])
                if 0 <= box[1]-previous[3] <= 1.8*height and same_column(previous, box, height):
                    combined.extend(second.pieces)
                    previous = box
                    if len(piece_lines(combined)) >= (4 if field == 'company_address' else 2):
                        break
            if len(combined) > len(first.pieces):
                candidates.append(Candidate(field, combined, 'model', reasons=['Aligned multi-line model span']))
    return candidates


def clean_model_candidate(candidate: Candidate, words: dict[int, Word]) -> None:
    """Strip key text using offsets, retaining measured character geometry."""
    pieces = candidate.pieces
    if not pieces:
        return
    if candidate.field == 'company_address':
        lines = piece_lines(pieces)
        while len(lines) > 1 and not plausible_address(stitch(lines[0], 'company_address')):
            lines.pop(0)
            candidate.reasons.append('Removed non-postal leading line from address span')
        candidate.pieces = pieces = [piece for line in lines for piece in line]
    row = Row([words[p.word_id] for p in pieces])
    # Candidate pieces are full OCR words at this stage, except heuristic candidates.
    anchors = anchors_for(row)
    if anchors and anchors[0]['start'] == 0:
        first = anchors[0]
        if first['field'] != candidate.field:
            candidate.pieces = []
            return
        candidate.pieces = select_value(row, first['end'], len(row.text), candidate.field)
        candidate.anchor, candidate.anchor_strength = first['text'], first['strength']
        candidate.reasons.append('Removed key text from model span')


def nearest_key(candidate: Candidate, rows: list[Row]) -> dict | None:
    box = union_box(p.bbox for p in candidate.pieces)
    options = []
    for row in rows:
        if vertical_overlap(box, row.bbox) < 0.5:
            continue
        for anchor in anchors_for(row):
            if anchor['bbox'][2] <= box[0] + row.height*0.2:
                options.append((max(0, box[0]-anchor['bbox'][2]), anchor))
    return min(options, key=lambda item: item[0])[1] if options else None


def inside_buyer_block(candidate: Candidate, rows: list[Row]) -> bool:
    box = union_box(p.bbox for p in candidate.pieces)
    for row in rows:
        for anchor in anchors_for(row):
            if anchor['field'] != 'buyer':
                continue
            key_box = anchor['bbox']
            if -row.height*0.3 <= box[1]-key_box[3] <= 7*row.height and abs(box[0]-key_box[0]) <= 2.5*row.height:
                return True
    return False


def rank_candidate(candidate: Candidate, rows: list[Row], predictions: dict[int, Prediction]) -> None:
    value = stitch(candidate.pieces, candidate.field)
    if not plausible(candidate.field, value):
        candidate.rank = -1.0
        candidate.reasons.append('Rejected: value syntax does not fit field')
        return
    model_score = weighted_score(candidate.pieces, predictions, candidate.field)
    ocr = ocr_score(candidate.pieces)
    if ocr < 0.35:
        candidate.rank = -1.0
        candidate.reasons.append('Rejected: mean OCR confidence below 0.35')
        return
    anchor = nearest_key(candidate, rows)
    if anchor:
        if (candidate.field in ('total', 'invoice_date', 'invoice_number') and anchor['field'] != candidate.field):
            candidate.rank = -1.0
            candidate.reasons.append('Rejected: nearest key belongs to ' + anchor['field'])
            return
        if anchor['field'] == candidate.field:
            candidate.anchor, candidate.anchor_strength = anchor['text'], anchor['strength']
    if candidate.field in ('vendor_name', 'company_address') and inside_buyer_block(candidate, rows):
        candidate.rank = -1.0
        candidate.reasons.append('Rejected: value lies in a labelled buyer block')
        return
    if candidate.field == 'vendor_name':
        for line in piece_lines(candidate.pieces):
            box = union_box(p.bbox for p in line)
            for row in rows:
                for segment in segments(row):
                    if (vertical_overlap(box, segment.bbox) >= 0.5 and
                            {p.word_id for p in line}.issubset({w.id for w in segment.words})
                            and plausible_address(segment.text)):
                        candidate.rank = -1.0
                        candidate.reasons.append('Rejected: name is a fragment of a postal-address line')
                        return
    candidate.rank = 0.55*model_score + 0.25*ocr + 0.20
    if candidate.anchor_strength:
        # A strong explicit key can repair a weak model; the result stays attributed.
        candidate.rank = max(candidate.rank, 0.58+0.18*candidate.anchor_strength+0.15*ocr+0.09*model_score)
    elif candidate.field in ('invoice_number', 'invoice_date', 'total'):
        candidate.rank = min(candidate.rank, 0.86)
    elif candidate.source == 'heuristic':
        candidate.rank = 0.57 + 0.20*ocr + 0.10*model_score
    if candidate.field == 'company_address' and len(piece_lines(candidate.pieces)) > 1:
        candidate.rank += 0.035
    candidate.rank = min(candidate.rank, 0.999)


def aggregate(words: list[Word], predictions: dict[int, Prediction], *, use_heuristics: bool = True,
              minimum_rank: float = 0.68) -> tuple[dict, dict]:
    rows = make_rows(words)
    word_map = {word.id: word for word in words}
    candidates = model_candidates(rows, predictions)
    for candidate in candidates:
        clean_model_candidate(candidate, word_map)
    if use_heuristics:
        candidates.extend(key_candidates(rows))
        # An unlabelled issuing-party header is considered only beside an address.
        image_bottom = max((w.bbox[3] for w in words), default=1)
        for row_index, row in enumerate(rows):
            if row.bbox[1] > image_bottom*0.32:
                break
            for segment in segments(row):
                if anchors_for(segment) or re.search(r'\b(?:invoice|receipt|quotation|purchase\s+order)\b', segment.text, re.I):
                    continue
                pieces = segment.pieces()
                if plausible('vendor_name', segment.text) and len(segment.words) >= 2 and address_after(pieces, rows):
                    candidates.append(Candidate('vendor_name', pieces, 'heuristic', reasons=['Header adjacent to a plausible postal address']))
                elif plausible('vendor_name', segment.text) and row_index+1 < len(rows):
                    following = rows[row_index+1]
                    if 0 <= following.bbox[1]-segment.bbox[3] <= 1.8*segment.height:
                        for continuation in segments(following):
                            combined = pieces+continuation.pieces()
                            if (same_column(segment.bbox,continuation.bbox,segment.height)
                                    and not anchors_for(continuation) and plausible('vendor_name',continuation.text)
                                    and address_after(combined,rows)):
                                candidates.append(Candidate('vendor_name',combined,'heuristic',
                                                            reasons=['Two aligned name lines followed by a postal address']))
        # Address continuation is anchored to a defensible seller-name candidate.
        for vendor in [c for c in candidates if c.field == 'vendor_name' and c.pieces]:
            rank_candidate(vendor, rows, predictions)
            if vendor.rank >= minimum_rank:
                address = address_after(vendor.pieces, rows)
                if address:
                    candidates.append(Candidate('company_address', address, 'heuristic',
                                                reasons=['Postal lines directly below seller name']))
    candidates = [candidate for candidate in candidates if candidate.pieces]
    for candidate in candidates:
        rank_candidate(candidate, rows, predictions)
    # A complete, well-supported name/address should not lose merely because one
    # short fragment has a slightly higher mean probability. Only expand measured
    # spans already admitted by geometry and syntax; never add text here.
    for full in candidates:
        if full.field not in ('vendor_name','company_address') or full.rank < minimum_rank or ocr_score(full.pieces) < 0.80:
            continue
        full_ids = {p.word_id for p in full.pieces}
        for fragment in candidates:
            if (fragment.field == full.field and fragment.rank >= minimum_rank and
                    {p.word_id for p in fragment.pieces} < full_ids):
                full.rank = min(0.999,max(full.rank,fragment.rank+0.015))
                full.reasons.append('Preferred complete geometric span over its shorter fragment')
    fields = dict.fromkeys(FIELDS)
    trace = {'thresholds': {'model_seed': 0.50, 'ocr_word_floor': 0.15, 'ocr_field_floor': 0.35,
                            'minimum_rank': minimum_rank}, 'selected': {}, 'candidates': []}
    used: set[tuple[int, int, int]] = set()
    # Select structured values first; a word cannot silently fill two distinct fields.
    for field in ('invoice_number', 'invoice_date', 'total', 'vendor_name', 'company_address'):
        options = [candidate for candidate in candidates if candidate.field == field and candidate.rank >= minimum_rank]
        options.sort(key=lambda c: (-c.rank, -len(c.pieces), union_box(p.bbox for p in c.pieces)[1]))
        unique = []
        for candidate in options:
            identity = tuple((p.word_id, p.start, p.end) for p in candidate.pieces)
            if not any(identity == old[0] for old in unique):
                unique.append((identity, candidate))
        if not unique:
            continue
        identity, best = unique[0]
        if any((p.word_id == word_id and p.start < end and p.end > start)
               for p in best.pieces for word_id, start, end in used):
            trace['selected'][field] = {'decision': 'abstained: overlapping field values'}
            continue
        # Close independent model-only candidates are ambiguous; preserve null.
        if len(unique) > 1:
            runner = unique[1][1]
            ids_a = {p.word_id for p in best.pieces}
            ids_b = {p.word_id for p in runner.pieces}
            if best.rank-runner.rank < 0.03 and not ids_a.intersection(ids_b) and best.anchor_strength == runner.anchor_strength:
                trace['selected'][field] = {'decision': 'abstained: two similarly supported candidates'}
                continue
        used.update(identity)
        model_score = weighted_score(best.pieces, predictions, field)
        confidence = sum(max(1, len(p.text)) * p.ocr_confidence * predictions.get(p.word_id, Prediction()).scores.get(field, 0)
                         for p in best.pieces) / sum(max(1, len(p.text)) for p in best.pieces)
        # Heuristic confidence is an evidence score, explicitly not a calibrated probability.
        source = 'model' if best.source == 'model' and not best.anchor_strength else ('hybrid' if model_score >= 0.5 else 'heuristic')
        if source != 'model':
            confidence = ocr_score(best.pieces)*(0.55*model_score+0.45*(best.anchor_strength or 0.65))
        value = stitch(best.pieces, field)
        result = {'value': value, 'bbox': list(union_box(p.bbox for p in best.pieces)),
                  'confidence': round(confidence, 4), 'token_count': len({p.word_id for p in best.pieces})}
        lines = piece_lines(best.pieces)
        if len(lines) > 1:
            result['line_boxes'] = [list(union_box(p.bbox for p in line)) for line in lines]
        if field == 'invoice_date' and (normalized := normalize_date(value)):
            result['normalized'] = normalized
        if field == 'total':
            for symbol, code in [('₹', 'INR'), ('INR', 'INR'), ('USD', 'USD'), ('EUR', 'EUR'), ('€', 'EUR'), ('GBP', 'GBP'), ('£', 'GBP'), ('CAD', 'CAD'), ('AUD', 'AUD')]:
                if symbol in value:
                    result['currency'] = code
                    break
        fields[field] = result
        trace['selected'][field] = {'decision': 'selected', 'source': source, 'rank': round(best.rank, 4),
                                     'model_score': round(model_score, 4), 'anchor': best.anchor,
                                     'word_ids': sorted({p.word_id for p in best.pieces}),
                                     'approximate_box': any(p.approximate_box for p in best.pieces), 'reasons': best.reasons}
    for candidate in candidates:
        trace['candidates'].append({'field': candidate.field, 'value': stitch(candidate.pieces, candidate.field),
                                    'bbox': list(union_box(p.bbox for p in candidate.pieces)),
                                    'source': candidate.source, 'rank': round(candidate.rank, 4),
                                    'anchor': candidate.anchor, 'reasons': candidate.reasons})
    return fields, trace
