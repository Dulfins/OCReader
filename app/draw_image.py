from PIL import Image, ImageDraw, ImageFont
import numpy as np

FONT_SCALE = 0.2

def wrap_text(text, font, max_width, draw):
    words = text.split(" ")
    lines = []
    current = ""

    for word in words:
        test = current + (" " if current else "") + word
        w, h = draw.textbbox((0, 0), test, font=font)[2:]
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines

def fit_text_to_box(
    draw,
    text,
    font_path,
    box_width,
    box_height,
    max_font_size=42,
    min_font_size=10,
    line_spacing=1.2
):
    
    for font_size in range(int(max_font_size * FONT_SCALE), int(min_font_size * FONT_SCALE) - 1, -1):
        font = ImageFont.truetype(font_path, font_size)
        lines = wrap_text(text, font, box_width, draw)

        line_height = font_size * line_spacing
        total_height = len(lines) * line_height

        if total_height <= box_height:
            return font, lines, line_height

    # fallback (smallest font)
    font = ImageFont.truetype(font_path, min_font_size)
    lines = wrap_text(text, font, box_width, draw)
    return font, lines, min_font_size * line_spacing


def render_translation_on_image(
    image_np,
    boxes,
    translations,
    font_path,
    text_color=(0, 0, 0)
):
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)

    for box, text in zip(boxes, translations):
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        padding = int(0.08 * box_width)
        usable_width = box_width - 2 * padding
        usable_height = box_height - 2 * padding

        font, lines, line_height = fit_text_to_box(
            draw,
            text,
            font_path,
            usable_width,
            usable_height
        )

        total_text_height = len(lines) * line_height
        y = y1 + (box_height - total_text_height) / 2

        for line in lines:
            w, h = draw.textbbox((0, 0), line, font=font)[2:]
            x = x1 + (box_width - w) / 2
            draw.text((x, y), line, font=font, fill=text_color)
            y += line_height

    return img
