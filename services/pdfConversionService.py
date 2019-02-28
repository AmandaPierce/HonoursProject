import pdf2image
import os
from pdf2image import convert_from_path

class pdfConversionService:
         def __init__(self):
                 self.pdf = ''
                 
        def convertImage(self, pdf):
                pages = convert_from_path(pdf, 500)
                counter = 0
                for page in pages:
                        page.save(os.path.splitext(pdf)[0] + '_' + str(counter) + '.jpg', 'JPEG')
                        counter += 1

