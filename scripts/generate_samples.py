"""Deterministic synthetic fixtures with independently measured ground-truth boxes.

Coordinates describe fixture artwork, never extraction rules. Development layouts
are separate from held-out layouts. Stress cases expose explicit limitations.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('vendor_name', 'company_address', 'invoice_number', 'invoice_date', 'total')


class Invoice:
    def __init__(self, size=(1200, 1600), accent='#1f3a5f'):
        self.image = Image.new('RGB', size, 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.accent = accent
        self.fields = {field: [] for field in FIELDS}

    def text(self, x, y, value, size=25, field=None, bold=False, color='#17202a'):
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        face = ImageFont.truetype(str(ROOT / 'assets/fonts' / name), size)
        self.draw.text((x, y), value, font=face, fill=color)
        if field:
            bbox = list(self.draw.textbbox((x, y), value, font=face))
            self.fields[field].append({'value': value, 'bbox': bbox})

    def rule(self, y, left=70, right=None):
        self.draw.line((left, y, right or self.image.width-70, y), fill=self.accent, width=2)

    def table(self, y, width=None):
        right = width or self.image.width-75
        self.draw.rectangle((70, y, right, y+48), fill=self.accent)
        self.text(88, y+9, 'DESCRIPTION', 20, bold=True, color='white')
        self.text(right-310, y+9, 'QTY', 20, bold=True, color='white')
        self.text(right-180, y+9, 'AMOUNT', 20, bold=True, color='white')
        for index, (description, amount) in enumerate([('Implementation services', '2,000.00'), ('Support and maintenance', '450.00')]):
            yy = y+75+index*62
            self.text(88, yy, description, 23)
            self.text(right-290, yy, '1', 23)
            self.text(right-180, yy, amount, 23)
            self.rule(yy+48, 70, right)

    def save(self, directory, name, split, note='', rotate=0):
        directory.mkdir(parents=True, exist_ok=True)
        self.image.save(directory / f'{name}.png')
        truth = {}
        for field, lines in self.fields.items():
            if not lines:
                truth[field] = None
            else:
                boxes = [line['bbox'] for line in lines]
                truth[field] = {'value': '\n'.join(line['value'] for line in lines),
                                'bbox': [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                         max(b[2] for b in boxes), max(b[3] for b in boxes)],
                                'line_boxes': boxes}
        return {'file': f'{name}.png', 'split': split, 'note': note, 'rotate': rotate,
                'image_size': list(self.image.size), 'fields': truth}


def classic():
    inv = Invoice()
    inv.text(75, 65, 'Northstar Analytics Pvt. Ltd.', 32, 'vendor_name', True)
    inv.text(75, 125, '42 Crescent Road', field='company_address')
    inv.text(75, 161, 'Indore, Madhya Pradesh 452001', field='company_address')
    inv.text(75, 210, 'Email: billing@example.test', 20)
    inv.text(810, 65, 'INVOICE', 42, bold=True, color=inv.accent)
    inv.text(720, 265, 'Invoice Number:', 23)
    inv.text(720, 303, 'NS-2026-01842', 28, 'invoice_number', True)
    inv.text(720, 365, 'Invoice Date:', 23)
    inv.text(720, 403, '15/08/2026', 26, 'invoice_date')
    inv.rule(475)
    inv.text(75, 510, 'BILL TO', 22, bold=True)
    inv.text(75, 550, 'Mistral Retail Limited', 26)
    inv.text(75, 591, '8 Lake Avenue, Pune 411001', 23)
    inv.text(720, 550, 'Due Date: 30/08/2026', 22)
    inv.table(700)
    inv.text(720, 1030, 'Subtotal:', 25)
    inv.text(965, 1030, '2,450.00', 25)
    inv.text(720, 1080, 'Tax:', 25)
    inv.text(965, 1080, '441.00', 25)
    inv.rule(1130, 700, 1130)
    inv.text(720, 1160, 'Grand Total:', 26, bold=True)
    inv.text(915, 1220, 'INR 2,891.00', 27, 'total', True)
    inv.text(75, 1420, 'Payment terms: 15 days. Thank you for your business.', 20)
    return inv


def right_header():
    inv = Invoice((1200, 1500), '#26483d')
    inv.text(75, 70, 'INVOICE', 48, bold=True, color=inv.accent)
    inv.text(660, 70, 'Oakbridge Office Supplies', 30, 'vendor_name', True)
    inv.text(660, 125, '17 Willow Street', 25, 'company_address')
    inv.text(660, 161, 'Bristol BS1 4AA', 25, 'company_address')
    inv.text(660, 215, 'Tel: 0117 555 0100', 20)
    inv.rule(280)
    inv.text(75, 325, 'Invoice No: OB/2026/0719', 27)
    # Measure just the value, exactly where the full line renders it.
    face = ImageFont.truetype(str(ROOT/'assets/fonts/DejaVuSans.ttf'), 27)
    x = 75 + inv.draw.textlength('Invoice No: ', font=face)
    inv.fields['invoice_number'].append({'value': 'OB/2026/0719', 'bbox': list(inv.draw.textbbox((x,325),'OB/2026/0719',font=face))})
    inv.text(75, 380, 'Date:', 25)
    inv.text(190, 380, '21 August 2026', 25, 'invoice_date')
    inv.text(75, 485, 'Customer:', 23, bold=True)
    inv.text(75, 530, 'Hawthorn Learning Centre', 26)
    inv.text(75, 570, '92 Mill Road, Bath BA1 1AB', 24)
    inv.text(660, 530, 'Delivery Date: 23/08/2026', 22)
    inv.table(680)
    inv.text(75, 985, 'Sub Total:', 25)
    inv.text(360, 985, 'GBP 2,450.00', 25)
    inv.text(75, 1030, 'VAT:', 25)
    inv.text(360, 1030, 'GBP 490.00', 25)
    inv.text(75, 1110, 'Invoice Total:', 27, bold=True)
    inv.text(360, 1110, 'GBP 2,940.00', 27, 'total', True)
    inv.text(75, 1190, 'Paid:', 24)
    inv.text(360, 1190, 'GBP 1,000.00', 24)
    inv.text(75, 1240, 'Balance Due:', 24)
    inv.text(360, 1240, 'GBP 1,940.00', 24)
    inv.text(75, 1390, 'Please quote the invoice number when paying.', 20)
    return inv


def ledger():
    inv = Invoice((1400, 1650), '#4b344b')
    inv.text(90, 65, 'CEDAR & FINCH DESIGN STUDIO', 35, 'vendor_name', True)
    inv.text(90, 125, 'Suite 4, 280 Market Street', 25, 'company_address')
    inv.text(90, 161, 'San Francisco, CA 94103', 25, 'company_address')
    inv.text(90, 220, 'www.example.test', 20)
    inv.text(970, 80, 'TAX INVOICE', 36, bold=True)
    inv.rule(290)
    inv.text(90, 340, 'INVOICE NUMBER', 23, bold=True)
    inv.text(90, 385, 'CFD-1048-A', 29, 'invoice_number')
    inv.text(630, 340, 'DATE OF ISSUE', 23, bold=True)
    inv.text(630, 385, '2026-08-24', 29, 'invoice_date')
    inv.text(90, 500, 'Billed To:', 24, bold=True)
    inv.text(90, 545, 'Seabrook Publishing Inc.', 26)
    inv.text(90, 590, '14 Union Street, Oakland CA 94612', 24)
    inv.table(720)
    inv.text(820, 1050, 'Subtotal', 26)
    inv.text(1090, 1050, '2,450.00', 26)
    inv.text(820, 1100, 'Sales Tax', 26)
    inv.text(1090, 1100, '196.00', 26)
    inv.rule(1160, 800, 1320)
    inv.text(820, 1200, 'Total Amount', 27, bold=True)
    inv.text(1090, 1260, 'USD 2,646.00', 27, 'total', True)
    inv.text(90, 1490, 'Services completed. Payment within 30 days.', 21)
    return inv


def heldout_sidebar():
    inv = Invoice((1200, 1500), '#174f63')
    inv.text(65, 60, 'INVOICE', 42, bold=True)
    inv.text(65, 155, 'Invoice #', 24)
    inv.text(65, 196, 'AT-982/26', 28, 'invoice_number')
    inv.text(65, 270, 'Issued On', 24)
    inv.text(65, 311, '26 August 2026', 25, 'invoice_date')
    inv.text(530, 75, 'Supplier:', 22, bold=True)
    inv.text(530, 116, 'Atlas Technical Services', 32, 'vendor_name', True)
    inv.text(530, 175, 'Building 9, 18 Station Road', 25, 'company_address')
    inv.text(530, 211, 'Leeds LS1 2AB', 25, 'company_address')
    inv.text(530, 260, 'Email: accounts@example.test', 21)
    inv.text(530, 365, 'Ship To:', 23, bold=True)
    inv.text(530, 410, 'Pinewood Research Ltd.', 25)
    inv.text(530, 449, '6 Grove Street, York YO1 7HP', 23)
    inv.table(640)
    inv.text(720, 1000, 'Subtotal', 25)
    inv.text(980, 1000, '2,450.00', 25)
    inv.text(720, 1060, 'Tax', 25)
    inv.text(980, 1060, '490.00', 25)
    inv.text(720, 1140, 'Amount Payable', 27, bold=True)
    inv.text(720, 1190, 'GBP 2,940.00', 29, 'total', True)
    return inv


def heldout_centered():
    inv = Invoice((1100, 1500), '#5d442c')
    inv.text(260, 70, 'MAPLE CREEK CONSULTING', 33, 'vendor_name', True)
    inv.text(260, 130, '61 Harbour Avenue', 25, 'company_address')
    inv.text(260, 167, 'Toronto, ON M5J 2N8', 25, 'company_address')
    inv.rule(250)
    inv.text(75, 300, 'Invoice ID:', 25)
    inv.text(305, 300, 'MC-2026-557', 27, 'invoice_number')
    inv.text(75, 360, 'Issue Date:', 25)
    inv.text(305, 360, '2026/08/27', 27, 'invoice_date')
    inv.text(75, 470, 'Sold To:', 24)
    inv.text(75, 515, 'Silver Fern Labs', 28)
    inv.text(75, 558, '88 King Street, Ottawa ON K1A 0B1', 22)
    inv.table(700)
    inv.text(75, 1080, 'Total', 30, bold=True)
    inv.text(470, 1080, 'CAD 2,768.50', 30, 'total', True)
    inv.text(75, 1150, 'Due Date: 2026/09/26', 24)
    return inv


def stress_cases():
    # Failures are observed by actually running OCR/model later, never fabricated predictions.
    faint = classic()
    total = faint.fields['total'][0]
    box = total['bbox']
    faint.draw.rectangle((box[0]-4,box[1]-4,box[2]+4,box[3]+4), fill='white')
    faint.fields['total'] = []
    faint.text(915, 1220, 'INR 2,891.00', 27, 'total', True, color='#eeeeee')
    dense = classic()
    dense.draw.rectangle((65,65,680,215),fill='white')
    dense.fields['vendor_name'], dense.fields['company_address'] = [], []
    dense.text(75,65,'NORTHSTAR',32,'vendor_name',True)
    dense.text(75,106,'ANALYTICS PRIVATE LIMITED',27,'vendor_name',True)
    dense.text(75,154,'42 Crescent Road',25,'company_address')
    dense.text(75,190,'Indore, Madhya Pradesh 452001',25,'company_address')
    glyph = ledger()
    box = glyph.fields['total'][0]['bbox']
    glyph.draw.rectangle((box[0]-4,box[1]-4,box[2]+4,box[3]+4), fill='white')
    glyph.fields['total'] = []
    glyph.text(1090,1260,'₹2,646.00',27,'total',True)
    return [('faint_total',faint,'Very faint printed total tests OCR loss; null is preferable to guessing.'),
            ('wrapped_vendor',dense,'Two-line legal name tests separation from the address below.'),
            ('rupee_symbol',glyph,'Currency glyph recognition by the English OCR model.')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development','heldout','stress'], default='development')
    args = parser.parse_args()
    if args.split == 'development':
        cases = [('classic',classic(), 'Left issuer, key-above-value metadata, INR total.'),
                 ('right_header',right_header(), 'Right issuer, left buyer, partial payment and balance due.'),
                 ('ledger',ledger(), 'Wide ledger, uppercase issuer, separate metadata columns.')]
        directory = ROOT/'samples/input'
    elif args.split == 'heldout':
        cases = [('sidebar',heldout_sidebar(),'Reserved layout: right supplier block and left metadata sidebar.'),
                 ('centered',heldout_centered(),'Reserved layout: centered issuer and left-aligned total.')]
        directory = ROOT/'samples/heldout/input'
    else:
        cases = stress_cases()
        directory = ROOT/'samples/stress/input'
    manifest = [invoice.save(directory,name,args.split,note) for name,invoice,note in cases]
    if args.split == 'stress':
        blank = Invoice()
        manifest.append(blank.save(directory,'blank','stress','Blank page: every field must be null.'))
        missing = classic()
        missing.draw.rectangle((700,1140,1145,1300), fill='white')
        missing.fields['total'] = []
        manifest.append(missing.save(directory,'missing_total','stress','Grand total absent; subtotal and tax remain.'))
    for case in manifest:
        for result in case['fields'].values():
            if result:
                result['bbox'] = [int(round(v)) for v in result['bbox']]
                result['line_boxes'] = [[int(round(v)) for v in b] for b in result['line_boxes']]
    manifest_path = directory.parent/'ground_truth.json'
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Generated {len(manifest)} {args.split} invoices: {directory}')


if __name__ == '__main__':
    main()
