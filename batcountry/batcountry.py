"""The 'bat-country' package is an easy to use, highly extendible, lightweight
Python module for inceptionism and deep dreaming with Convolutional Neural
Networks and Caffe."""

from __future__ import division, print_function

import logging
import os
import tempfile
import time

from google.protobuf import text_format
import numpy as np
from PIL import Image

os.environ['GLOG_minloglevel'] = '1'
import caffe


class Stopwatch:
    def __init__(self):
        self.last = time.perf_counter()
        self.total = 0.0

    def delta(self):
        now = time.perf_counter()
        delta = now - self.last
        self.last = now
        self.total += delta
        return delta


class BatCountry:
    """Contains all BatCountry functionality."""
    def __init__(self, base_path='', deploy_path=None, model_path=None,
                 mean=(104.0, 117.0, 123.0), channels=(2, 1, 0), device=None,
                 logger=None):
        # None: default device; -1: CPU; n for n >= 0: GPU n
        if device is not None:
            if device < 0:
                caffe.set_mode_cpu()
            else:
                caffe.set_device(device)
                caffe.set_mode_gpu()

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(format='%(message)s')

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
              verbose=True, visualize=False, progress=None, **step_params):
        # if a step function has not been supplied, initialize it as the
        # standard gradient ascent step
        if step_fn is None:
            step_fn = BatCountry.gradient_ascent_step

        # if the objective function has not been supplied, initialize it
        # as the L2 objective
        if objective_fn is None:
            objective_fn = BatCountry.l2_objective

        # if the preprocess function has not been supplied, initialize it
        if preprocess_fn is None:
            preprocess_fn = BatCountry.preprocess

        # if the deprocess function has not been supplied, initialize it
        if deprocess_fn is None:
            deprocess_fn = BatCountry.deprocess

        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # initialize the visualization list
        visualizations = []

        self.logger.debug('dream() starting.')
        timer = Stopwatch()

        # prepare base images for all octaves
        octaves = [preprocess_fn(self.net, image)]
        h, w = octaves[0].shape[-2:]

        for i in range(1, octave_n):
            h_new = np.int32(np.round(h/octave_scale**i))
            w_new = np.int32(np.round(w/octave_scale**i))
            octaves.append(BatCountry.resize(octaves[0], h_new, w_new,
                                             Image.LANCZOS))

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
                detail = BatCountry.resize(detail, h, w)

            # resize the network's input image size
            src.reshape(1, 3, h, w)
            src.data[0] = octave_base + detail

            np.random.seed(seed)

            self.logger.debug('delta=%.3fs, preprocessing done.', timer.delta())

            for i in range(iter_n):
                step_fn(self.net, end=end, clip=clip,
                        objective_fn=objective_fn, **step_params)

                # visualization
                if visualize:
                    vis = deprocess_fn(self.net, src.data[0])

                    # adjust image contrast if clipping is disabled
                    if not clip:
                        vis = vis * (255.0 / np.percentile(vis, 99.98))

                self.logger.debug('delta=%.3fs, octave=%d, iter=%d, layer=%s, image_dim=%s',
                                  timer.delta(), octave, i, end, src.data[0].shape[::-1])

                if progress is not None:
                    progress(octave, i)

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

        self.logger.debug('delta=%.3fs, dream() completed in %.2fs.', timer.delta(), timer.total)
        return r

    @staticmethod
    def gradient_ascent_step(net, step_size=1.5, end='inception_4c/output',
                             jitter=32, clip=True, objective_fn=None,
                             **objective_params):
        # if the objective function is None, initialize it as
        # the standard L2 objective
        if objective_fn is None:
            objective_fn = BatCountry.l2_objective

        # input image is stored in Net's 'data' blob
        src = net.blobs['data']
        dst = net.blobs[end]

        # apply jitter shift
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        src_f = src.data[0]
        src.data[0, ...] = np.roll(np.roll(src_f, ox, -1), oy, -2)

        # for blob in net.blobs.values():
        #     blob.diff[:] = 0
        net.forward(end=end)
        objective_fn(dst, **objective_params)
        net.backward(start=end)
        g = src.diff[0]

        # apply normalized ascent step to the input image
        src_f = src.data[0] + step_size / np.abs(g).mean() * g

        # unshift image
        src_f = np.roll(np.roll(src_f, -ox, -1), -oy, -2)

        # unshift image
        if clip:
            bias = net.transformer.mean['data']
            src_f = np.clip(src_f, -bias, 255 - bias)
        src.data[0, ...] = src_f

    def layers(self):
        # return the layers of the network
        layers = []
        for i, layer in enumerate(self.net.blobs.keys()):
            if i == 0:
                continue
            if layer.find('_split_') == -1:
                layers.append(layer)
        return layers

    def prepare_guide(self, image, end='inception_4c/output',
                      preprocess_fn=None):
        # if the preprocess function has not been supplied, initialize it
        if preprocess_fn is None:
            preprocess_fn = BatCountry.preprocess

        # grab dimensions of input image
        (w, h) = image.size

        # GoogLeNet was trained on images with maximum width and heights
        # of 224 pixels -- if either dimension is larger than 224 pixels,
        # then we'll need to do some resizing
        n_w, n_h = 224, 224
        if w != 224 or h != 224:
            image = np.float32(image.resize((n_w, n_h), Image.LANCZOS))
        else:
            image = np.float32(image)

        (src, dst) = (self.net.blobs['data'], self.net.blobs[end])
        src.reshape(1, 3, n_h, n_w)
        src.data[0] = preprocess_fn(self.net, image)
        self.net.forward(end=end)
        guide_features = dst.data[0].copy()

        return guide_features

    @staticmethod
    def l2_objective(dst):
        dst.diff[0, ...] = dst.data[0]

    @staticmethod
    def guided_objective(dst, objective_features):
        x = dst.data[0].copy()
        y = objective_features
        ch = x.shape[0]
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)

        # compute the matrix of dot-products with guide features
        a = x.T.dot(y)

        # select ones that match best
        dst.diff[0].reshape(ch, -1)[:] = y[:, a.argmax(1)]

    @staticmethod
    def preprocess(net, img):
        return np.float32(
            np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

    @staticmethod
    def deprocess(net, img):
        return np.dstack((img + net.transformer.mean['data'])[::-1])

    @staticmethod
    def resize(arr, h, w, method=Image.BICUBIC):
        arr = np.float32(arr)
        if arr.ndim == 3:
            planes = [arr[i, :, :] for i in range(arr.shape[0])]
        else:
            raise TypeError('Only 3D CxHxW arrays are supported')
        imgs = [Image.fromarray(plane) for plane in planes]
        imgs_resized = [img.resize((w, h), method) for img in imgs]
        return np.stack([np.array(img) for img in imgs_resized])
