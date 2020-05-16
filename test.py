import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import time


image = face_recognition.load_image_file("Obama_Merkel.jpg")
face_locations = face_recognition.face_locations(image)
print(face_locations)
print("Haha")

pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left) in face_locations:
    draw.rectangle(((left, top), (right, bottom)), outline=(0,0,255))
pil_image.show()

time.sleep(5)
