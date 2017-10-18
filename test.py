import moxel

model = moxel.Model('moxel/awesome:latest', where='localhost')

image = moxel.space.Image.from_file('volkswagen.jpg')

result = model.predict(image=image)

print('result', result)
