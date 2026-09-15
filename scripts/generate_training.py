"""Create 36 training and 8 validation invoices, with randomized content/geometry.

These invoice identities are disjoint from the supplied development and held-out
fixtures. This remains a small synthetic distribution, not a real-world benchmark.
"""
import argparse
import json
import random
from pathlib import Path

from generate_samples import Invoice, ROOT

NAMES = ['Juniper Systems Limited','Bluewater Engineering Ltd.','Copperleaf Media Studio',
         'Ridgeway Logistics Pvt. Ltd.','Orchid Industrial Services','Lighthouse Digital Works',
         'Granite Peak Solutions','Willowbrook Trading Company','Kestrel Office Products',
         'Evergreen Research Partners','Meridian Creative Agency','Harborline Software Ltd.']
STREETS = ['18 Westbridge Road','72 Orchard Street','5 East Park Avenue','Building 3, 64 Station Lane',
           'Suite 12, 45 Commercial Road','109 Cedar Drive']
CITIES = ['Mumbai, Maharashtra 400001','Bengaluru, Karnataka 560001','London EC1A 1BB',
          'Manchester M1 1AE','Austin, TX 78701','Boston, MA 02108']


def make_invoice(rng, index):
    width = rng.choice([1200,1300,1400])
    inv = Invoice((width,1600),rng.choice(['#1f3a5f','#26483d','#4b344b']))
    variant = index % 4
    x = rng.randint(55,100) if variant != 1 else width//2+rng.randint(20,55)
    y = rng.randint(45,80)
    size = rng.randint(25,31)
    inv.text(x,y,rng.choice(NAMES),size,'vendor_name',True)
    inv.text(x,y+57,rng.choice(STREETS),23,'company_address')
    inv.text(x,y+93,rng.choice(CITIES),23,'company_address')
    inv.text(x,y+142,'Email: billing@example.test',19)
    title_x = width-310 if variant != 1 else 70
    inv.text(title_x,75,'INVOICE',39,bold=True)
    inv.rule(270)
    number = rng.choice([f'JN-{2024+index%3}-{rng.randint(1000,9999)}',f'INV/{rng.randint(100,999)}/26',f'{rng.randint(100000,999999)}'])
    date = rng.choice([f'{rng.randint(13,28):02d}/08/2026', f'2026-07-{rng.randint(13,28):02d}',f'{rng.randint(13,28)} August 2026'])
    key_x = rng.randint(65,95)
    if variant in (0,1):
        inv.text(key_x,325,rng.choice(['Invoice Number:','Invoice No:','Invoice ID:']),24)
        inv.text(key_x+270,325,number,26,'invoice_number')
        inv.text(key_x,385,rng.choice(['Invoice Date:','Issue Date:','Date:']),24)
        inv.text(key_x+270,385,date,26,'invoice_date')
    else:
        inv.text(key_x,310,'Invoice Number',23,bold=True)
        inv.text(key_x,355,number,27,'invoice_number')
        inv.text(width//2,310,'Date of Issue',23,bold=True)
        inv.text(width//2,355,date,27,'invoice_date')
    inv.text(75,475,rng.choice(['Bill To:','Customer:','Billed To:']),23,bold=True)
    inv.text(75,520,'Elmwood Distribution Ltd.',26)
    inv.text(75,560,'91 Hill Street, Cambridge CB1 1AB',23)
    inv.text(width//2+40,520,'Due Date: 30/09/2026',22)
    inv.table(670)
    subtotal = rng.randint(100,9000)+rng.choice([0.0,0.5,0.25])
    tax = round(subtotal*0.18,2)
    total = subtotal+tax
    currency = rng.choice(['USD','INR','GBP','EUR','CAD'])
    tx = width-530 if variant != 3 else 75
    inv.text(tx,1015,'Subtotal:',25)
    inv.text(tx+260,1015,f'{subtotal:,.2f}',25)
    inv.text(tx,1070,'Tax:',25)
    inv.text(tx+260,1070,f'{tax:,.2f}',25)
    key = rng.choice(['Grand Total:','Invoice Total:','Total Amount:','Total:'])
    inv.text(tx,1150,key,25,bold=True)
    if variant in (0,3):
        inv.text(tx+260,1150,f'{currency} {total:,.2f}',25,'total',True)
    else:
        inv.text(tx+120,1210,f'{currency} {total:,.2f}',27,'total',True)
    inv.text(75,1410,'Payment terms: 30 days. Thank you for your business.',20)
    return inv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/training-data')
    parser.add_argument('--seed',type=int,default=1729)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    for split,count in [('train',36),('validation',8)]:
        directory = args.output_dir/split/'input'
        cases = [make_invoice(rng,i).save(directory,f'{split}_{i:03}',split) for i in range(count)]
        (directory.parent/'ground_truth.json').write_text(json.dumps(cases,indent=2)+'\n',encoding='utf-8')
    print(f'Generated 36 training and 8 validation invoices, seed={args.seed}: {args.output_dir}')


if __name__ == '__main__':
    main()
