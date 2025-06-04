#!/usr/bin/env python3
"""
make_full_bleed_charuco.py

Embed “gridboard.png” into a PDF sized exactly 90 cm wide (height scaled to preserve aspect ratio),
with no margins (full‐bleed). The resulting PDF page will be width=90 cm and height as needed.
"""
import os
# Switch to this script's parent directory to find the Charuco board image.
os.chdir(os.path.dirname(__file__))
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from PIL import Image

# 1) Load the PNG to get its pixel dimensions
input_png = "gridboard.png"
im = Image.open(input_png)
w_px, h_px = im.size
aspect_ratio = h_px / w_px  # height/width

# 2) Desired printed width: 90 cm
width_cm = 90.0
width_pts = width_cm * cm  # ReportLab: 1 cm = 28.3464567 points

# 3) Calculate the necessary height in points to preserve aspect ratio
height_pts = width_pts * aspect_ratio

# 4) Create a PDF canvas with page size (90 cm × computed height)
output_pdf = "charuco_90cm_full_bleed.pdf"
c = canvas.Canvas(output_pdf, pagesize=(width_pts, height_pts))

# 5) Draw the image at (0,0) filling the entire page
c.drawImage(input_png, 0, 0, width=width_pts, height=height_pts)

# 6) Finalize and save
c.showPage()
c.save()

print(f"Loaded “{input_png}” ({w_px}×{h_px} px, aspect ratio ≈ {aspect_ratio:.3f})")
print(f"PDF page will be {width_cm} cm wide × {height_pts/cm:.2f} cm tall")
print(f"Saved “{output_pdf}” with no margins (full‐bleed).")
