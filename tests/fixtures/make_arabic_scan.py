import io
from pathlib import Path

import arabic_reshaper
import fitz
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import random

LINES = [
    "تقرير الإيرادات السنوية لعام 2025",
    "نمت الإيرادات بنسبة اثني عشر بالمئة",
    "وبلغ صافي الربح ثلاثة ملايين درهم",
]

def shape(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))

font = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 46)
width, height = 1654, 900
img = Image.new("L", (width, height), 255)
draw = ImageDraw.Draw(img)
for i, line in enumerate(LINES):
    text = shape(line)
    w = draw.textlength(text, font=font)
    draw.text((width - w - 120, 120 + i * 150), text, font=font, fill=20)

img = img.rotate(0.6, expand=False, fillcolor=255).filter(ImageFilter.GaussianBlur(0.6))
rng = random.Random(7)
px = img.load()
for _ in range(9000):
    x, y = rng.randrange(width), rng.randrange(height)
    px[x, y] = rng.randint(120, 200)

buf = io.BytesIO()
img.save(buf, format="PNG")
doc = fitz.open()
page = doc.new_page(width=595, height=int(595 * height / width) + 0)
page.insert_image(page.rect, stream=buf.getvalue())
out = Path(__file__).with_name("arabic_scan.pdf")
doc.save(str(out), deflate=True)
print("wrote", out, out.stat().st_size, "bytes")
