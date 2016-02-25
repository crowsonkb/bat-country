from __future__ import division, print_function

import os
import sys
import tempfile

import caffe
from google.protobuf import text_format
import numpy as np
from PIL import Image
from skimage.transform import rescale, resize

class BatCountry:
    def __init__(self, base_path, deploy_path=None, model_path=None,
                 mean=(104.0, 117.0, 123.0), channels=(2, 1, 0)):
        # if the deploy path is None, set the default
        if deploy_path is None:
            deploy_path = base_path + '/deploy.prototxt'
        else:
            deploy_path = base_path + deploy_path

        # if the model path is None, set it to the default GoogleLeNet model
        if model_path is None:
            model_path = base_path + '/bvlc_googlenet.caffemodel'
        else:
            model_path = base_path + model_path

        # patch the model to compute gradients
        model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(deploy_path).read(), model)
        model.force_backward = True
        patch_tmp = tempfile.NamedTemporaryFile(
            mode='w+', suffix='.prototxt', delete=False)
        patch_tmp.file.write(str(model))
        patch_tmp.file.close()

        # load the network and store the patched model path
        self.net = caffe.Classifier(patch_tmp.name, model_path,
                                    mean=np.float32(mean),
                                    channel_swap=channels)
        self.patch_model = patch_tmp.name

    def dream(self, image, iter_n=10, octave_n=4, octave_scale=np.sqrt(2),
              end='inception_4c/output', clip=True, seed=0, step_fn=None,
              objective_fn=None, preprocess_fn=None, deprocess_fn=None,
              verbose=True, visualize=False, **step_params):
        # if a step function has not been supplied, initialize it as the
        # standard gradient ascent step
        if step_fn is None:
            step_fn = BatCountry.gradient_ascent_step

        # if the objective function has not been supplied, initialize it
        # as the L2 objective
        if objective_fn is None:
            objective_fn = BatCountry.L2_objective

        # if the preprocess function has not been supplied, initialize it
        if preprocess_fn is None:
            preprocess_fn = BatCountry.preprocess

        # if the deprocess function has not been supplied, initialize it
        if deprocess_fn is None:
            deprocess_fn = BatCountry.deprocess

        # initialize the visualization list
        visualizations = []

        # prepare base images for all octaves
        octaves = [preprocess_fn(self.net, image)]

        for i in range(octave_n - 1):
            octaves.append(rescale(
                octaves[-1].T/255, 1/octave_scale, order=3).T*255)

        # allocate image for network-produced details
        detail = np.zeros_like(octaves[-1])
        src = self.net.blobs['data']

        # clear all blob gradients
        for blob in self.net.blobs.values():
            blob.diff[:] = 0

        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]

            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = resize(detail.T/255, (w, h, 3), order=3).T*255

            # resize the network's input image size
            src.reshape(1, 3, h, w)
            src.data[0] = octave_base + detail

            np.random.seed(seed)
            for i in range(iter_n):
                step_fn(self.net, end=end, clip=clip,
                        objective_fn=objective_fn, **step_params)

                # visualization
                if visualize:
                    vis = deprocess_fn(self.net, src.data[0])

                    # adjust image contrast if clipping is disabled
                    if not clip:
                        vis = vis * (255.0 / np.percentile(vis, 99.98))

                if verbose:
                    print('octave={}, iter={}, layer={}, image_dim={}'.format(
                          octave, i, end, src.data[0].shape))
                    sys.stdout.flush()

                # check to see if the visualization list should be
                # updated
                if visualize:
                    k = 'octave_{}-iter_{}-layer_{}'.format(
                        octave, i, end.replace('/', '_'))
                    visualizations.append((k, vis))

            # extract details produced on the current octave
            detail = src.data[0] - octave_base

        # grab the resulting image
        r = deprocess_fn(self.net, src.data[0])

        # check to see if the visualizations should be included
        if visualize:
            r = (r, visualizations)

        return r

    @staticmethod
    def gradient_ascent_step(net, step_size=1.5, end='inception_4c/output',
                             jitter=32, clip=True, objective_fn=None,
                             **objective_params):
        # if the objective function is None, initialize it as
        # the standard L2 objective
        if objective_fn is None:
            objective_fn = BatCountry.L2_objective

        # input image is stored in Net's 'data' blob
        src = net.blobs['data']
        dst = net.blobs[end]

        # apply jitter shift
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)

        for blob in net.blobs.values():
            blob.diff[:] = 0
        net.forward(end=end)
        objective_fn(dst, **objective_params)
        net.backward(start=end)
        g = src.diff[0]

        # apply normalized ascent step to the input image
        src.data[:] += step_size / np.abs(g).mean() * g

        # unshift image
        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)

        # unshift image
        if clip:
            bias = net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255 - bias)

    def layers(self):
        # return the layers of the network
        return self.net._layer_names

    def cleanup(self):
        # remove the patched model from disk
        os.remove(self.patch_model)

    def prepare_guide(self, image, end='inception_4c/output',
                      maxW=224, maxH=224, preprocess_fn=None):
        # if the preprocess function has not been supplied, initialize it
        if preprocess_fn is None:
            preprocess_fn = BatCountry.preprocess

        # grab dimensions of input image
        (w, h) = image.size

        # GoogLeNet was trained on images with maximum width and heights
        # of 224 pixels -- if either dimension is larger than 224 pixels,
        # then we'll need to do some resizing
        nW, nH = 224, 224
        if w != 224 or h != 224:
            image = np.float32(image.resize((nW, nH), Image.LANCZOS))
        else:
            image = np.float32(image)

        (src, dst) = (self.net.blobs['data'], self.net.blobs[end])
        src.reshape(1, 3, nH, nW)
        src.data[0] = preprocess_fn(self.net, image)
        self.net.forward(end=end)
        guide_features = dst.data[0].copy()

        return guide_features

    @staticmethod
    def L2_objective(dst):
        dst.diff[:] = dst.data

    @staticmethod
    def guided_objective(dst, objective_features):
        x = dst.data[0].copy()
        y = objective_features
        ch = x.shape[0]
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)

        # compute the matrix of dot-products with guide features
        A = x.T.dot(y)

        # select ones that match best
        dst.diff[0].reshape(ch, -1)[:] = y[:, A.argmax(1)]

    @staticmethod
    def preprocess(net, img):
        return np.float32(
            np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

    @staticmethod
    def deprocess(net, img):
        return np.dstack((img + net.transformer.mean['data'])[::-1])
