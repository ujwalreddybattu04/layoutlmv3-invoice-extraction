"""Train one encoder block plus a six-label head; save only the adapted tensors.

Freezing the other blocks keeps this practical on a 6 GB GPU. The small adapter
requires the pinned base checkpoint; it is not a standalone model checkpoint.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast

from src.layout import readable_order
from src.merge import normalize_box
from src.model import DEFAULT_MODEL, DEFAULT_REVISION
from src.ocr import TesseractOCR, load_image
from src.types import Word

LABELS = ['O','VENDOR_NAME','VENDOR_ADDR','INVOICE_NUM','INVOICE_DATE','TOTAL']
FIELD_ID = {'vendor_name':1,'company_address':2,'invoice_number':3,'invoice_date':4,'total':5}


def labels_for(words, truth):
    labels = []
    for word in words:
        x1,y1,x2,y2 = word.bbox
        best = (0.0,0)
        for field,result in truth.items():
            if result is None:
                continue
            for b in result['line_boxes']:
                overlap = max(0,min(x2,b[2])-max(x1,b[0]))*max(0,min(y2,b[3])-max(y1,b[1])) / max(1,(x2-x1)*(y2-y1))
                if overlap > best[0]:
                    best = (overlap,FIELD_ID[field])
        labels.append(best[1] if best[0]>=0.70 else 0)
    return labels


def prepare(manifest, ocr, tokenizer, image_processor):
    records = []
    for case in json.loads(manifest.read_text(encoding='utf-8')):
        path = manifest.parent/'input'/case['file']
        image = load_image(path)
        words = readable_order(ocr(image))
        labels = labels_for(words,case['fields'])
        encoded = tokenizer([w.text for w in words],boxes=[normalize_box(w.bbox,image.size) for w in words],
                            word_labels=labels, padding='max_length',max_length=512,truncation=True,return_tensors='pt')
        encoded['pixel_values'] = image_processor(image,return_tensors='pt')['pixel_values']
        records.append({k:v for k,v in encoded.items()})
    return records


def score(model, records, device):
    model.eval()
    confusion = torch.zeros((6,6),dtype=torch.long)
    with torch.inference_mode():
        for record in records:
            inputs = {key:value.to(device) for key,value in record.items()}
            target = inputs.pop('labels')
            output = model(**inputs).logits.argmax(-1)
            mask = target != -100
            for expected,actual in zip(target[mask].cpu().tolist(),output[mask].cpu().tolist()):
                confusion[expected,actual]+=1
    f1 = []
    for label in range(1,6):
        tp = int(confusion[label,label])
        denominator = int(confusion[label,:].sum()+confusion[:,label].sum())
        f1.append(2*tp/denominator if denominator else 0.0)
    return {'macro_f1_fields':sum(f1)/len(f1),'per_field_f1':dict(zip(LABELS[1:],f1)),
            'confusion':confusion.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('artifacts/training-data'))
    parser.add_argument('--base-model',default=DEFAULT_MODEL)
    parser.add_argument('--output',type=Path,default=Path('assets/adaptation'))
    parser.add_argument('--tesseract-cmd')
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    started = time.perf_counter()
    options = {'revision':DEFAULT_REVISION} if args.base_model == DEFAULT_MODEL else {'local_files_only':True}
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(args.base_model,**options)
    image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
    ocr = TesseractOCR(args.tesseract_cmd)
    print('Preparing OCR-aligned training data...',flush=True)
    train = prepare(args.data/'train/ground_truth.json',ocr,tokenizer,image_processor)
    validation = prepare(args.data/'validation/ground_truth.json',ocr,tokenizer,image_processor)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.base_model, num_labels=6,id2label=dict(enumerate(LABELS)),label2id={v:k for k,v in enumerate(LABELS)},
        ignore_mismatched_sizes=True,use_safetensors=True,**options).to(args.device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith('classifier.') or name.startswith('layoutlmv3.encoder.layer.11.')
    names = [name for name,parameter in model.named_parameters() if parameter.requires_grad]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('classifier.')],'lr':2e-5},
                                   {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('classifier.')],'lr':5e-4}],weight_decay=0.01)
    # Mild class weights keep the O-heavy page from dominating small fields.
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.35,1,1,1.4,1.4,1.4],device=args.device),ignore_index=-100)
    scaler = torch.amp.GradScaler('cuda',enabled=args.device=='cuda')
    print(f'Trainable parameters: {trainable}; train={len(train)}, validation={len(validation)}',flush=True)
    history=[]
    best=-1
    args.output.mkdir(parents=True,exist_ok=True)
    for epoch in range(args.epochs):
        model.eval()  # frozen backbone stays deterministic; train dropout only in adapted block/head
        model.layoutlmv3.encoder.layer[11].train()
        model.classifier.train()
        order=list(range(len(train))); random.shuffle(order)
        losses=[]
        for index in order:
            record={key:value.to(args.device) for key,value in train[index].items()}
            target=record.pop('labels')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device,dtype=torch.float16,enabled=args.device=='cuda'):
                logits=model(**record).logits
                loss=loss_fn(logits.reshape(-1,6),target.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        metrics=score(model,validation,args.device)
        row={'epoch':epoch+1,'mean_training_loss':sum(losses)/len(losses),**metrics}
        history.append(row)
        print(f'Epoch {epoch+1}: loss={row["mean_training_loss"]:.4f}; validation field token macro-F1={metrics["macro_f1_fields"]:.4f}',flush=True)
        if metrics['macro_f1_fields']>best:
            best=metrics['macro_f1_fields']
            state=model.state_dict()
            save_file({name:state[name].detach().cpu().contiguous() for name in names},str(args.output/'weights.safetensors'))
            metadata={'base_model':DEFAULT_MODEL,'base_revision':DEFAULT_REVISION,
                      'base_weights_sha256':'a6be323e1fe46aac11bc3a34055828513479a07cc8a8172dfd2574ae87a1374c',
                      'labels':LABELS,'label_scheme':'flat','trainable_parameters':trainable,
                      'trained_tensor_names':names,'selected_epoch':epoch+1,'seed':args.seed,
                      'train_invoices':len(train),'validation_invoices':len(validation),'validation':metrics}
            (args.output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    report={'elapsed_sec':round(time.perf_counter()-started,2),'device':args.device,'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,
            'epochs':args.epochs,'seed':args.seed,'history':history,'best_validation_field_token_macro_f1':best}
    (args.output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Finished in {report["elapsed_sec"]:.1f}s; saved adapted tensors in {args.output}',flush=True)


if __name__ == '__main__':
    main()
