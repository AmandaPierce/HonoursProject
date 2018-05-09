import PythonMagick
from PythonMagick import Image

def convertImage(pdf):
    img = Image()
    # canvas resolution
    img.density('300')
    img.read(pdf)
    size = "%sx%s" % (img.columns(), img.rows())
    output_img = Image(size, "#ffffff")
    output_img.type = img.type
    output_img.composite(img, 0, 0, PythonMagick.CompositeOperator.SrcOverCompositeOp)
    output_img.magick('JPG')
    output_img.quality(100)
    jpg = pdf.replace(".pdf", ".jpg")
    return jpg
