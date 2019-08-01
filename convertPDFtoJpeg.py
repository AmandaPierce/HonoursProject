from pdf2image import convert_from_path


def convertImage(pdf):
    pages = convert_from_path(pdf, 500)
    counter = 0
    for page in pages:
        page.save('images/exampleE.jpg', 'JPEG')
        counter += 1
