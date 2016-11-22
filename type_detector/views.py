from django.shortcuts import render
# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.http import HttpResponse
from django.core import serializers
from scipy import misc # To load/save images without Caffe
import numpy as np
import urllib
import json
import cv2
import os
import matplotlib.pyplot as plt
import h5py # to save/load data files
import sys
import caffe

images_path = 'type_detector/libs'

labels = [] # Initialising labels as an empty array.
home_dir = os.getenv("HOME")
caffe_root = os.path.join(home_dir, 'Git/caffe')
sys.path.insert(0, os.path.join(caffe_root, 'python'))
model_def = os.path.join(caffe_root, 'models', 'bvlc_reference_caffenet','deploy.prototxt')
model_weights = os.path.join(caffe_root, 'models','bvlc_reference_caffenet','bvlc_reference_caffenet.caffemodel')
net = caffe.Net(model_def,model_weights,caffe.TEST)

mu = np.load(os.path.join(caffe_root, 'python','caffe','imagenet','ilsvrc_2012_mean.npy'))
mu = mu.mean(1).mean(1)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
labels_file = os.path.join(caffe_root, 'data','ilsvrc12','synset_words.txt')
if not os.path.exists(labels_file):
	os.system("~/caffe/data/ilsvrc12/get_ilsvrc_aux.sh")
labels = np.loadtxt(labels_file, str, delimiter='\t')


def detect(request):

	if os.path.isfile(os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')):
		print 'CaffeNet found.'
	else:
		print 'Downloading pre-trained CaffeNet model...'
		os.system("~/caffe/scripts/download_model_binary.py ~/caffe/models/bvlc_reference_caffenet")

	print 'mean-subtracted values:', zip('BGR', mu)

	print "-------------------"
	print "-------------------"

	KNN = NearestNeighbors(images_path=images_path)
	vectors, img_files = load_dataset_hist(images_path)
	KNN.setXtr(vectors)

	# Freeing memory:
	del vectors
	KNN.setFilesList(img_files)

	my_image_url = "https://upload.wikimedia.org/wikipedia/commons/b/be/Orang_Utan%2C_Semenggok_Forest_Reserve%2C_Sarawak%2C_Borneo%2C_Malaysia.JPG"
	#my_image_url = "http://www.pyimagesearch.com/wp-content/uploads/2015/05/obama.jpg"
	urllib.urlretrieve (my_image_url, "image.jpg")
	image =  misc.imread('image.jpg')

	vectors, img_files = load_dataset(images_path)
	KNN = NearestNeighbors(Xtr=vectors, img_files=img_files, images_path=images_path, labels=labels)

	del vectors
	print "-------------------"
	print "-------------------"

	data = {'122':'sedr'}

	KNN.retrieve(predict_imageNet('image.jpg'))
	#return JsonResponse(data)
	print "-------------------"
	print "-------------------"

	return render(request, 'type_detector/detect.html')

def predict_imageNet(image_filename):
    image = caffe.io.load_image(image_filename)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)

    # perform classification
    net.forward()
	# obtain the output probabilities
    output_prob = net.blobs['prob'].data[0]
    # sort top ten predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:10]

    print 'probabilities and labels:'
    predictions = (output_prob[top_inds], labels[top_inds]) # showing only labels (skipping the index)

    for p in predictions:
        print p
    return output_prob


def load_dataset(images_path):
    vectors_filename = os.path.join(images_path, 'vectors.h5')

    if os.path.exists(vectors_filename):
        print 'Loading image signatures (probability vectors) from ' + vectors_filename
        with h5py.File(vectors_filename, 'r') as f:
            vectors = f['vectors'][()]
            img_files = f['img_files'][()]

    else:
        # Build a list of JPG files (change if you want other image types):
        os.listdir(images_path)
        img_files = [f for f in os.listdir(images_path) if (('jpg' in f) or ('JPG') in f)]

        print 'Loading all images to the memory and pre-processing them...'

        net_data_shape = net.blobs['data'].data.shape
        train_images = np.zeros(([len(img_files)] + list(net_data_shape[1:])))

        for (f,n) in zip(img_files, range(len(img_files))):
            print '%d %s'% (n,f)
            image = caffe.io.load_image(os.path.join(images_path, f))
            train_images[n] = transformer.preprocess('data', image)

        print 'Extracting descriptor vector (classifying) for all images...'
        vectors = np.zeros((train_images.shape[0],1000))
        for n in range(0,train_images.shape[0],10): # For each batch of 10 images:
            # This block can/should be parallelised!
            print 'Processing batch %d' % n
            last_n = np.min((n+10, train_images.shape[0]))

            net.blobs['data'].data[0:last_n-n] = train_images[n:last_n]

            # perform classification
            net.forward()

            # obtain the output probabilities
            vectors[n:last_n] = net.blobs['prob'].data[0:last_n-n]

        print 'Saving descriptors and file indices to ' + vectors_filename
        with h5py.File(vectors_filename, 'w') as f:
            f.create_dataset('vectors', data=vectors)
            f.create_dataset('img_files', data=img_files)

    return vectors, img_files

def get_hist(filename):
    image =  misc.imread(filename)
    image = image[::4,::4,:]
    im_norm = (image-image.mean())/image.std()

    hist_red = np.histogram(im_norm[:,:,0], range=(-np.e,+np.e))[0]
    hist_green = np.histogram(im_norm[:,:,1], range=(-np.e,+np.e))[0]
    hist_blue = np.histogram(im_norm[:,:,2], range=(-np.e,+np.e))[0]
    # Concatenating them into a 30-dimensional vector:
    histogram = np.concatenate((hist_red, hist_green, hist_blue)).astype(np.float)
    return histogram/histogram.sum()

def load_dataset_hist(images_path):
    vectors_filename = os.path.join(images_path, 'vectors_hist.h5')

    if os.path.exists(vectors_filename):
        print 'Loading image signatures (colour histograms) from ' + vectors_filename
        with h5py.File(vectors_filename, 'r') as f:
            vectors = f['vectors'][()]
            img_files = f['img_files'][()]

    else:
        # Build a list of JPG files (change if you want other image types):
        os.listdir(images_path)
        img_files = [f for f in os.listdir(images_path) if (('jpg' in f) or ('JPG') in f)]

        print 'Loading all images to the memory and pre-processing them...'
        vectors = np.zeros((len(img_files), 30))
        for (f,n) in zip(img_files, range(len(img_files))):
            print '%d %s'% (n,f)
            vectors[n] = get_hist(os.path.join(images_path, f))

        print 'Saving descriptors and file indices to ' + vectors_filename
        with h5py.File(vectors_filename, 'w') as f:
            f.create_dataset('vectors', data=vectors)
            f.create_dataset('img_files', data=img_files)

    return vectors, img_files

class NearestNeighbors:
    def __init__(self, K=10, Xtr=[], images_path='Photos/', img_files=[], labels=np.empty(0)):
        # Setting defaults
        self.K = K
        self.Xtr = Xtr
        self.images_path = images_path
        self.img_files = img_files
        self.labels = labels

    def setXtr(self, Xtr):
        """ X is N x D where each row is an example."""
        self.Xtr = Xtr

    def setK(self, K):
        """ K is the number of samples to be retrieved for each query."""
        self.K = K

    def setImagesPath(self,images_path):
        self.images_path = images_path

    def setFilesList(self,img_files):
        self.img_files = img_files

    def setLabels(self,labels):
        self.labels = labels

    def predict(self, x):
        """ x is a test (query) sample vector of 1 x D dimensions """
        distances = np.sum(np.abs(self.Xtr-x), axis = 1)
        return np.argsort(distances) # returns an array of indices of of the samples, sorted by how similar they are to x.

    def retrieve(self, x):
        nearest_neighbours = self.predict(x)
