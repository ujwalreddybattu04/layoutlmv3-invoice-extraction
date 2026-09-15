"""A side legend keeps field labels completely outside the invoice text."""
from PIL import Image, ImageDraw, ImageFont

from .types import FIELDS

COLORS = {'vendor_name': '#166534', 'company_address': '#1d4ed8', 'invoice_number': '#a21caf',
          'invoice_date': '#b45309', 'total': '#b91c1c'}


def font(size: int):
    return ImageFont.load_default(size=size)


def wrap_text(text: str, draw, face, width: int) -> list[str]:
    lines = []
    for paragraph in text.splitlines() or ['']:
        line = ''
        for char in paragraph:
            if line and draw.textlength(line+char, font=face) > width:
                lines.append(line)
                line = ''
            line += char
        lines.append(line)
    return lines


def annotate(image: Image.Image, fields: dict, *, mode: str) -> Image.Image:
    scale = max(1.0, min(2.0, image.width/1000))
    sidebar = int(360*scale)
    margin = int(22*scale)
    body, heading = font(int(15*scale)), font(int(18*scale))
    line_height = int(22*scale)
    measuring = ImageDraw.Draw(image)
    blocks = []
    for field in FIELDS:
        result = fields[field]
        value = result['value'] if result else 'Not detected'
        lines = wrap_text(value, measuring, body, sidebar-2*margin)
        blocks.append((field, result, lines))
    needed_height = int(125*scale) + sum((len(lines)+3)*line_height for _, _, lines in blocks)
    canvas = Image.new('RGB', (image.width+sidebar, max(image.height, needed_height)), '#f3f4f6')
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    x, y = image.width+margin, margin
    draw.text((x, y), 'INVOICE EXTRACTION', fill='#111827', font=heading)
    y += line_height*1.5
    draw.text((x, y), f'Mode: {mode}', fill='#4b5563', font=body)
    y += line_height*2
    for field, result, lines in blocks:
        color = COLORS[field] if result else '#6b7280'
        if result:
            draw.rectangle(result['bbox'], outline=color, width=max(2, int(3*scale)))
        draw.rectangle((x, y+3, x+10*scale, y+13*scale), fill=color)
        draw.text((x+18*scale, y), field.replace('_', ' ').upper(), fill=color, font=body)
        y += line_height*1.3
        for line in lines:
            draw.text((x, y), line, fill='#111827' if result else '#6b7280', font=body)
            y += line_height
        if result:
            draw.text((x, y), f"Evidence confidence: {result['confidence']:.2f}", fill='#4b5563', font=body)
            y += line_height
        y += line_height
    return canvas

