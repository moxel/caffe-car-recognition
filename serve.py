import caffe
import caffe.io
import numpy as np
import moxel

net = caffe.Classifier('deploy.prototxt', 'googlenet_finetune_web_car.caffemodel',
                           image_dims=(224, 224))
with open('cars.txt', 'r') as f:
    labels = [line.replace('\n', '') for line in f.readlines()]

def predict(image):
    image = image.to_numpy_rgb()[:, :, :3]
    image = np.array(image, dtype='float32')
    pred = net.predict([image])[0]
    result = labels[pred.argmax()]

    return {
        'model': result
    }
