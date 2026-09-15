from src.aggregate import aggregate
from src.types import Prediction, Word


def test_table_quantity_cannot_become_invoice_number():
    words=[Word(0,'1',(100,300,110,320),0.99)]
    predictions={0:Prediction('B-INVOICE_NUM',0.999,{'invoice_number':0.999})}
    fields,_=aggregate(words,predictions)
    assert fields['invoice_number'] is None


def test_footer_stop_word_cannot_become_vendor():
    words=[Word(0,'for',(100,300,130,320),0.99)]
    fields,_=aggregate(words,{0:Prediction('B-VENDOR_NAME',0.99,{'vendor_name':0.99})})
    assert fields['vendor_name'] is None


def test_explicit_invoice_key_beats_unkeyed_model_number():
    words=[Word(0,'Invoice',(10,10,80,30),0.99),Word(1,'Number:',(90,10,160,30),0.99),
           Word(2,'INV-123',(200,10,280,30),0.99),Word(3,'987654',(200,500,260,520),0.99)]
    fields,_=aggregate(words,{3:Prediction('B-INVOICE_NUM',0.999,{'invoice_number':0.999})})
    assert fields['invoice_number']['value']=='INV-123'


def test_decimal_amount_is_not_an_invoice_identifier():
    words=[Word(0,'2,000.00',(100,300,190,320),0.99)]
    fields,_=aggregate(words,{0:Prediction('B-INVOICE_NUM',0.99,{'invoice_number':0.99})})
    assert fields['invoice_number'] is None


def test_uncertain_connector_between_name_words_is_retained():
    words=[Word(0,'Cedar',(10,20,60,40),0.98),Word(1,'&',(70,20,80,40),0.98),
           Word(2,'Finch',(90,20,140,40),0.98)]
    predictions={0:Prediction('VENDOR_NAME',0.9,{'vendor_name':0.9}),
                 1:Prediction('VENDOR_ADDR',0.75,{'company_address':0.75,'vendor_name':0.2}),
                 2:Prediction('VENDOR_NAME',0.95,{'vendor_name':0.95})}
    fields,trace=aggregate(words,predictions,use_heuristics=False)
    assert fields['vendor_name']['value']=='Cedar & Finch'
    assert 'Bridged one uncertain interior word' in trace['selected']['vendor_name']['reasons']


def test_phone_line_cannot_poison_complete_address_candidate():
    words=[Word(0,'17 River Road',(10,20,140,40),0.98),Word(1,'Bristol BS1 4AA',(10,55,150,75),0.98),
           Word(2,'Tel: 555 0100',(10,100,140,120),0.98)]
    predictions={i:Prediction('VENDOR_ADDR',0.99,{'company_address':0.99}) for i in range(3)}
    fields,_=aggregate(words,predictions,use_heuristics=False)
    assert fields['company_address']['value']=='17 River Road\nBristol BS1 4AA'


def test_street_fragment_cannot_become_vendor():
    words=[Word(0,'61',(10,50,30,70),0.99),Word(1,'Harbour',(40,50,110,70),0.99),
           Word(2,'Avenue',(120,50,180,70),0.99)]
    fields,_=aggregate(words,{1:Prediction('VENDOR_NAME',0.99,{'vendor_name':0.99})})
    assert fields['vendor_name'] is None


def test_wrapped_name_is_complete_and_separate_from_address():
    words=[Word(0,'HORIZON',(10,10,120,40),0.99),Word(1,'RESEARCH LIMITED',(10,55,240,80),0.99),
           Word(2,'42 River Road',(10,100,200,125),0.99),Word(3,'Leeds LS1 2AB',(10,140,200,165),0.99)]
    fields,_=aggregate(words,{1:Prediction('VENDOR_ADDR',0.99,{'company_address':0.99}),
                             2:Prediction('VENDOR_ADDR',0.99,{'company_address':0.99}),
                             3:Prediction('VENDOR_ADDR',0.99,{'company_address':0.99})})
    assert fields['vendor_name']['value']=='HORIZON\nRESEARCH LIMITED'
    assert fields['company_address']['value']=='42 River Road\nLeeds LS1 2AB'
