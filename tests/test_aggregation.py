import pytest

from src.aggregate import aggregate, normalize_date
from src.types import FIELDS, Prediction, Word


def word(i, text, x, y, width=None, confidence=0.96):
    width = width or len(text)*10
    chars = tuple((x+j*10,y,x+(j+1)*10,y+20) for j in range(len(text)))
    return Word(i,text,(x,y,x+width,y+20),confidence,char_boxes=chars)


def predictions(words, labels):
    result = {}
    names = {'vendor_name': 'VENDOR_NAME', 'company_address': 'VENDOR_ADDR',
             'invoice_number': 'INVOICE_NUM', 'invoice_date': 'INVOICE_DATE', 'total': 'TOTAL'}
    for w, field in zip(words,labels):
        result[w.id] = Prediction('B-'+names[field],0.95,{field:0.95}) if field else Prediction()
    return result


def test_blank_returns_exactly_five_nulls():
    fields, _ = aggregate([], {})
    assert fields == dict.fromkeys(FIELDS)


def test_split_invoice_number_is_stitched_without_key():
    words = [word(0,'Invoice',10,20),word(1,'Number:',90,20),
             word(2,'INV-2026',190,20),word(3,'-00125',272,20)]
    result, _ = aggregate(words,predictions(words,[None,None,'invoice_number','invoice_number']))
    assert result['invoice_number']['value'] == 'INV-2026-00125'
    assert result['invoice_number']['bbox'] == [190,20,332,40]
    assert result['invoice_number']['token_count'] == 2


def test_fused_key_value_uses_real_character_boxes():
    words = [word(0,'Total:₹52,450.00',10,20)]
    result, trace = aggregate(words,{})
    assert result['total']['value'] == '₹52,450.00'
    assert result['total']['bbox'][0] == 70
    assert result['total']['currency'] == 'INR'
    assert trace['selected']['total']['approximate_box'] is False


def test_fused_key_without_char_boxes_does_not_fake_geometry():
    words = [Word(0,'Total:50.00',(10,20,160,40),0.99)]
    result, trace = aggregate(words,{})
    assert result['total']['value'] == '50.00'
    assert result['total']['bbox'] == [10,20,160,40]
    assert trace['selected']['total']['approximate_box'] is True


def test_multiline_address_keeps_lines_and_union():
    words = [word(0,'Cobalt',10,10),word(1,'Labs',85,10),
             word(2,'17',10,50),word(3,'River',40,50),word(4,'Road',100,50),
             word(5,'Bristol',10,85),word(6,'BS1',95,85),word(7,'4AA',135,85)]
    result, _ = aggregate(words,predictions(words,['vendor_name']*2+['company_address']*6))
    address = result['company_address']
    assert address['value'] == '17 River Road\nBristol BS1 4AA'
    assert address['line_boxes'] == [[10,50,140,70],[10,85,165,105]]
    assert address['bbox'] == [10,50,165,105]


def test_missing_total_does_not_select_subtotal_or_balance():
    words = [word(0,'Subtotal:',10,100),word(1,'100.00',200,100),
             word(2,'Balance',10,160),word(3,'Due:',90,160),word(4,'70.00',200,160)]
    result, trace = aggregate(words,predictions(words,[None,'total',None,None,'total']))
    assert result['total'] is None
    assert any('blocked_total' in ' '.join(c['reasons']) for c in trace['candidates'])


def test_address_does_not_append_damaged_contact_label_in_separate_segment():
    # Real reduced-resolution failure: "Emai" and the email are separated by
    # 38 pixels, enough to split an 8-10 px-high OCR row into two segments.
    words = [Word(0,'Northstar',(39,35,125,48),.96),
             Word(1,'Analytics',(131,35,213,51),.96),
             Word(2,'42 Crescent Road',(38,65,147,75),.96),
             Word(3,'Indore, Madhya Pradesh 452001',(39,83,239,95),.96),
             Word(4,'Emai',(38,107,59,115),.85),
             Word(5,'someone@example.test',(97,107,210,117),.24)]
    result,_ = aggregate(words,predictions(words,['vendor_name']*2+['company_address']*3+[None]))
    assert result['company_address']['value'] == '42 Crescent Road\nIndore, Madhya Pradesh 452001'
    assert result['company_address']['bbox'] == [38,65,239,95]


@pytest.mark.parametrize('tail,contact_x', [('India',400),('Pune 411001',100)])
def test_address_contact_guard_keeps_distant_columns_and_postal_lines(tail,contact_x):
    words = [Word(0,'Example',(10,10,60,20),.96),Word(1,'Limited',(65,10,110,20),.96),
             Word(2,'42 River Road',(10,35,90,45),.96),
             Word(3,tail,(10,60,60,70),.96),
             Word(4,'hello@example.test',(contact_x,60,contact_x+120,70),.96)]
    result,_=aggregate(words,predictions(words,['vendor_name']*2+['company_address']*2+[None]))
    assert result['company_address']['value'] == '42 River Road\n'+tail


def test_grand_total_beats_high_model_subtotal():
    words = [word(0,'Sub',10,100),word(1,'Total:',50,100),word(2,'100.00',230,100),
             word(3,'Grand',10,150),word(4,'Total:',70,150),word(5,'118.00',230,150)]
    result, _ = aggregate(words,predictions(words,[None,None,'total',None,None,None]))
    assert result['total']['value'] == '118.00'


def test_due_date_is_not_invoice_date():
    words = [word(0,'Due',10,20),word(1,'Date:',50,20),word(2,'21/08/2026',180,20)]
    result, _ = aggregate(words,predictions(words,[None,None,'invoice_date']))
    assert result['invoice_date'] is None


def test_low_confidence_key_value_abstains():
    words = [word(0,'Total:',10,20),word(1,'900.00',110,20,confidence=0.12)]
    result, _ = aggregate(words,{})
    assert result['total'] is None


def test_buyer_name_not_returned_as_vendor():
    words = [word(0,'Bill',10,20),word(1,'To:',60,20),
             word(2,'Customer',10,60),word(3,'Limited',100,60)]
    result, _ = aggregate(words,predictions(words,[None,None,'vendor_name','vendor_name']))
    assert result['vendor_name'] is None


def test_unrelated_columns_are_not_merged():
    words = [word(0,'Alpha',10,20),word(1,'Ltd',70,20),
             word(2,'Beta',500,20),word(3,'Ltd',550,20)]
    result, trace = aggregate(words,predictions(words,['vendor_name']*4),use_heuristics=False)
    assert result['vendor_name'] is None  # two equally supported independent sellers
    assert 'similarly supported' in trace['selected']['vendor_name']['decision']


def test_model_only_cannot_invent_from_a_key_without_predictions():
    words = [word(0,'Total:',10,20),word(1,'100.00',110,20)]
    result, _ = aggregate(words,{},use_heuristics=False)
    assert result['total'] is None


def test_amount_below_key_can_be_right_aligned():
    words = [word(0,'Grand',10,20),word(1,'Total:',70,20),word(2,'INR',180,65),word(3,'100.00',220,65)]
    result, _ = aggregate(words,{})
    assert result['total']['value'] == 'INR 100.00'


@pytest.mark.parametrize('value,expected', [
    ('15/08/2026','2026-08-15'), ('2026-08-15','2026-08-15'),
    ('21 August 2026','2026-08-21'), ('03/04/2026',None), ('31/02/2026',None),
])
def test_date_normalization_is_conservative(value,expected):
    assert normalize_date(value) == expected


def test_ambiguous_date_preserves_raw_string_without_normalized_key():
    words = [word(0,'Date:',10,20),word(1,'03/04/2026',100,20)]
    result, _ = aggregate(words,{})
    assert result['invoice_date']['value'] == '03/04/2026'
    assert 'normalized' not in result['invoice_date']


def test_bare_dollar_does_not_imply_usd():
    words = [word(0,'Total:',10,20),word(1,'$100.00',100,20)]
    result, _ = aggregate(words,{})
    assert 'currency' not in result['total']
