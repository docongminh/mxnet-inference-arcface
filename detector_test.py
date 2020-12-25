from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
import face_image
import face_preprocess

def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer, prefix, epoch):
  """
   __docstring__
  """
  print('loading', prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size, image_size))])
  model.set_params(arg_params, aux_params)
  return model

def imshow(image, boxs):


    # Start coordinate, here (5, 5)
    # represents the top left corner of rectangle
    print("boxs: ", boxs)
    start_point = (int(boxs[0]), int(boxs[1]))

    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (int(boxs[2]), int(boxs[3]))

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    img = cv2.rectangle(image, start_point, end_point, color, thickness)

    # Displaying the image
    return img

class FaceModel:
  def __init__(self, args):
    self.args = args
    if args.use_cpu:
      ctx = mx.cpu()
    else:
      ctx = mx.gpu(args.gpu)
    self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector


  def get_input(self, face_img):
    """

    """
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    print("----------------------------------------------------------------")
    print(ret)
    print("----------------------------end---------------------------------")
    if ret is None:
      return None
    bbox, points = ret
    print("BOX: ", bbox)
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    print("--------------box--------------------")
    print(bbox)
    result = imshow(face_img, bbox)
    cv2.imwrite("test.jpg", result)
    print("---------------points---------------")
    print(points)
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned

if __name__ == '__main__':
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
    model = FaceModel(args)
    img = cv2.imread('/home/cristiand/Documents/face_detection/mxnet-inference-arcface/test/test_1.jpg')
    img = model.get_input(img)
