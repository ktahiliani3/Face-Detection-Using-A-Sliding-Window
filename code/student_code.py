import numpy as np
import cyvlfeat as vlfeat
from utils import *
import os.path as osp
from glob import glob
from random import shuffle
from IPython.core.debugger import set_trace
from sklearn.svm import LinearSVC
import cv2
from sklearn import tree


def get_positive_features(train_path_pos, feature_params):
    """

    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    positive_files = glob(osp.join(train_path_pos, '*.jpg'))

    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats = np.random.rand(len(positive_files), n_cell*n_cell*31)


    for i in range(len(positive_files)):

        Image = load_image_gray(positive_files[i])

        HogFeatures = vlfeat.hog.hog(Image, cell_size)

        feats[i, :] = HogFeatures.flatten()


    return feats

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.

    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))


    n_cell = np.ceil(win_size/cell_size).astype('int')
    D = n_cell*n_cell*31
    Features = np.zeros((num_samples,D))

    TotalImages = len(negative_files)
    s = 0

    for i in range(TotalImages):

        Image = load_image_gray(negative_files[i])
        Rows = Image.shape[0]
        Columns = Image.shape[1]

        for j in range(0,Rows - win_size,6):
            for k in range(0,Columns - win_size,6):

                if s == num_samples:
                    break
                ImageCrop = Image[j:j+win_size, k: k + win_size]
                HogFeatures = vlfeat.hog.hog(ImageCrop, cell_size)


                Features[s, : ] = np.ravel(HogFeatures)
                s = s + 1






    feats = Features

    return feats

def train_classifier(features_pos, features_neg, C):
    """

    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """

    svm = LinearSVC(random_state=0, tol=0.0000007, loss='hinge', C=C, max_iter = 1000000000)
    Features = np.vstack((features_pos, features_neg))
    Labels = np.hstack((np.ones(len(features_pos)), -np.ones(len(features_neg))))
    svm.fit(Features, Labels)


    return svm

def mine_hard_negs(non_face_scn_path, svm, feature_params):
    """

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object

    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))


    n_cell = np.ceil(win_size/cell_size).astype('int')
    D = n_cell*n_cell*31
    Features = np.zeros((2000,D))
    w = svm.coef_
    b = svm.intercept_


    TotalImages = len(negative_files)
    s = 0

    for i in range(TotalImages):

        Image = load_image_gray(negative_files[i])
        Rows = Image.shape[0]
        Columns = Image.shape[1]

        for j in range(0,Rows - win_size,13):
            for k in range(0,Columns - win_size,13):

                if s == 2000:
                    break
                ImageCrop = Image[j:j+win_size, k: k + win_size]
                HogFeatures = vlfeat.hog.hog(ImageCrop, cell_size)
                HogFeatures = np.ravel(HogFeatures)

                score = np.dot(HogFeatures,w.T) + b

                if score > -1:

                    Features[s, : ] = np.ravel(HogFeatures)
                    s = s + 1



    feats = Features


    return feats

def run_detector(test_scn_path, svm, feature_params, verbose=False):
    """

    Args:
    -   test_scn_path: (string) This directory contains images which may or
            may not have faces in them. This function should work for the
            MIT+CMU test set but also for any other images (e.g. class photos).
    -   svm: A trained sklearn.svm.LinearSVC object
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slowerbecause the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time.
    -   verbose: prints out debug information if True

    Returns:
    -   bboxes: N x 4 numpy array. N is the number of detections.
            bboxes(i,:) is [x_min, y_min, x_max, y_max] for detection i.
    -   confidences: (N, ) size numpy array. confidences(i) is the real-valued
            confidence of detection i.
    -   image_ids: List with N elements. image_ids[i] is the image file name
            for detection i. (not the full path, just 'albert.jpg')
    """
    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    image_ids = []

    # number of top detections to feed to NMS
    topk = 500
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    scale_factor = feature_params.get('scale_factor', 0.65)
    template_size = int(win_size / cell_size)

    for idx, im_filename in enumerate(im_filenames):
        print('Detecting faces in {:s}'.format(im_filename))
        im = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = im.shape
        # create scale space HOG pyramid and return scores for prediction

        scales = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1, 0.05, 0.025]
        cur_bboxes = np.empty((0, 4))
        cur_confidences = np.empty(0)
        for j in range(len(scales)):



            D = template_size*template_size*31
            threshold = -2

            Image = im
            ImageRows = Image.shape[0]
            ImageColumns = Image.shape[1]
            Image = cv2.resize(Image, None, fx = scales[j],fy = scales[j])



            Features = vlfeat.hog.hog(Image, cell_size)

            Rows = Features.shape[0]
            Columns = Features.shape[1]

            for k in range(Rows - template_size):
                for l in range(Columns - template_size):

                    ImageHog = Features[k:k + template_size, l:l+template_size, :]
                    HogReshape = np.ravel(ImageHog)
                    w = svm.coef_
                    b = svm.intercept_

                    score = np.dot(HogReshape,w.T) + b



                    if score > threshold:
                        y_min = (k)*cell_size
                        x_min = (l)*cell_size
                        y_max = y_min + win_size
                        x_max = x_min + win_size
                        y_min = np.floor(y_min/scales[j]) + 1
                        x_min = np.floor(x_min/scales[j]) + 1
                        y_max = np.floor(y_max/scales[j]) + 1
                        x_max = np.floor(x_max/scales[j]) + 1

                        BoundingBox = [int(x_min), int(y_min), int(x_max), int(y_max)]

                        cur_bboxes = np.vstack((cur_bboxes,BoundingBox))
                        cur_confidences = np.append(cur_confidences, score)












        ### non-maximum suppression ###

        idsort = np.argsort(-cur_confidences)[:topk]
        cur_bboxes = cur_bboxes[idsort]
        cur_confidences = cur_confidences[idsort]

        is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
            im_shape, verbose=verbose)

        print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))
        cur_bboxes = cur_bboxes[is_valid_bbox]
        cur_confidences = cur_confidences[is_valid_bbox]

        bboxes = np.vstack((bboxes, cur_bboxes))
        confidences = np.hstack((confidences, cur_confidences))
        image_ids.extend([im_id] * len(cur_confidences))

    return bboxes, confidences, image_ids

'''


def decision_tree_classifier(features_pos, features_neg):

    DecisionTree = tree.DecisionTreeClassifier()
    Features = np.vstack((features_pos, features_neg))
    Labels = np.hstack((np.ones(len(features_pos)), -np.ones(len(features_neg))))
    DecisionTree.fit(Features, Labels)

    return DecisionTree


def run_detector_tree(test_scn_path, decison_tree, feature_params, verbose=False):


    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    image_ids = []

    # number of top detections to feed to NMS
    topk = 500
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    scale_factor = feature_params.get('scale_factor', 0.65)
    template_size = int(win_size / cell_size)

    for idx, im_filename in enumerate(im_filenames):
        print('Detecting faces in {:s}'.format(im_filename))
        im = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = im.shape
        # create scale space HOG pyramid and return scores for prediction

        scales = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1, 0.05, 0.025]
        cur_bboxes = np.empty((0, 4))
        cur_confidences = np.empty(0)
        for j in range(len(scales)):



            D = template_size*template_size*31
            threshold = -1.1

            Image = im
            ImageRows = Image.shape[0]
            ImageColumns = Image.shape[1]
            Image = cv2.resize(Image, None, fx = scales[j],fy = scales[j])



            Features = vlfeat.hog.hog(Image, cell_size)

            Rows = Features.shape[0]
            Columns = Features.shape[1]

            for k in range(Rows - template_size):
                for l in range(Columns - template_size):

                    ImageHog = Features[k:k + template_size, l:l+template_size, :]
                    HogReshape = np.ravel(ImageHog)
                    HogReshape = np.reshape(HogReshape,(1,1116))



                    score = decison_tree.apply(HogReshape)



                    if score > threshold:
                        y_min = (k)*cell_size
                        x_min = (l)*cell_size
                        y_max = y_min + win_size
                        x_max = x_min + win_size
                        y_min = np.floor(y_min/scales[j]) + 1
                        x_min = np.floor(x_min/scales[j]) + 1
                        y_max = np.floor(y_max/scales[j]) + 1
                        x_max = np.floor(x_max/scales[j]) + 1

                        BoundingBox = [int(x_min), int(y_min), int(x_max), int(y_max)]

                        cur_bboxes = np.vstack((cur_bboxes,BoundingBox))
                        cur_confidences = np.append(cur_confidences, score)


        ### non-maximum suppression ###


        idsort = np.argsort(-cur_confidences)[:topk]
        cur_bboxes = cur_bboxes[idsort]
        cur_confidences = cur_confidences[idsort]

        is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
            im_shape, verbose=verbose)

        print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))
        cur_bboxes = cur_bboxes[is_valid_bbox]
        cur_confidences = cur_confidences[is_valid_bbox]

        bboxes = np.vstack((bboxes, cur_bboxes))
        confidences = np.hstack((confidences, cur_confidences))
        image_ids.extend([im_id] * len(cur_confidences))

    return bboxes, confidences, image_ids

'''


