import face_model
import argparse
import cv2
import sys
import time
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--prefix', default='models/arc_models/arc_resnet50/model-r50/model', type=str, help='prefix to model')
parser.add_argument('--epoch', default=0, type=int, help='epoch index')
parser.add_argument('--image_size', default=112, type=int, help='image size')
parser.add_argument('--use_cpu', default=True, type=bool, help='cpu mxnet')
args = parser.parse_args()
print("==============")
t1 = time.time()
model = face_model.FaceModel(args)

print("time init model: ", time.time()-t1)
t2 = time.time()
img = cv2.imread('images/minh_1.jpg')
img = model.get_input(img)
print("time detect: ", time.time()-t2)
t3 = time.time()
f2 = model.get_feature(img)
print(f2.shape)
print("time extract: ", time.time()-t3)