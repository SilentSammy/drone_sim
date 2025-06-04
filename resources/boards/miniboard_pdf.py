#!/usr/bin/env python3
"""
make_a4_charuco.py

Embed “gridboard.png” into an A4‐sized PDF, scaling it so that it prints exactly 9 cm wide,
with the height scaled to maintain the original aspect ratio. The image will be centered
horizontally and vertically on the A4 page, leaving typical A4 margins around it.
"""
import os
# Switch to this script's parent directory to find the Charuco board image.
os.chdir(os.path.dirname(__file__))
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from PIL import Image

# 1) Load the PNG to get its pixel dimensions
input_png = "gridboard.png"
im = Image.open(input_png)
w_px, h_px = im.size
aspect_ratio = h_px / w_px  # height/width

# 2) Desired printed width: 9 cm
width_cm = 9.0
width_pts = width_cm * cm  # ReportLab: 1 cm = 28.3464567 points

# 3) Calculate the height in points to preserve aspect ratio
height_pts = width_pts * aspect_ratio

# 4) A4 page dimensions in points
page_width, page_height = A4  # (595.2756, 841.8898) points

# 5) Compute coordinates to center the image on A4
x = (page_width - width_pts) / 2
y = (page_height - height_pts) / 2

# 6) Create PDF canvas
output_pdf = "charuco_on_a4.pdf"
c = canvas.Canvas(output_pdf, pagesize=A4)

# 7) Draw the image at computed size and position
c.drawImage(input_png, x, y, width=width_pts, height=height_pts)

# 8) Finalize and save
c.showPage()
c.save()

print(f"Loaded “{input_png}” ({w_px}×{h_px} px, aspect ratio ≈ {aspect_ratio:.3f})")
print(f"PDF page size: {page_width:.1f}×{page_height:.1f} points  (A4)")
print(f"Image will be {width_cm} cm wide × {height_pts/cm:.2f} cm tall on paper")
print(f"Saved “{output_pdf}” (image centered on A4).")
