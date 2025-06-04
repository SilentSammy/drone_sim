from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

from PIL import Image

# switch to this script's parent directory to find the Charuco board image.
import os
os.chdir(os.path.dirname(__file__))


# 1) Open your high-res Charuco PNG. Assume it's e.g. 10629×(height_in_px) at 300 DPI.
im = Image.open("gridboard.png")

# 2) Make sure PIL knows its DPI (so it embeds the right physical size)
#    If your image is already 300 DPI when you exported it, this may be embedded.
#    Otherwise, force it here:
desired_dpi = 300  # change if your print shop wants 600 DPI, etc.
im.info['dpi'] = (desired_dpi, desired_dpi)

# 3) Save as PDF, preserving that DPI → physical size mapping
#    Pillow will create a PDF page exactly sized so that (pixels / DPI) = inches on paper.
im.save("charuco_board.pdf", "PDF", resolution=desired_dpi)


# 1) Create a canvas that’s 90 cm wide and, say, 120 cm tall
c = canvas.Canvas("gridboard.pdf",
                  pagesize=(90*cm, 60*cm))  # width, height in points

# 2) Draw your image to fill the page (from (0,0) up to (90 cm, height))
c.drawImage("gridboard.png", 
            x=0, y=0,
            width=90*cm,              # 90 cm on paper
            height=(im.height / im.width) * 90 * cm)  # preserve aspect ratio

# 3) Finish
c.showPage()
c.save()
