from pdf2image import convert_from_path
#NIE DZIAŁA W IDE, TRZEBA W TERMINALU
images = convert_from_path("data/data.pdf", dpi=300)
for i, image in enumerate(images):
    image.save(f"page_{i}.png", "PNG")