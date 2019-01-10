#Face Detection Using A Sliding Window


Face detection has always been an active topic of research in the computer vision community. In this project, I have implemented a face detection algorithm based on sliding windows. The sliding window model classifies image patches as being an object or non-object, which in this case is a face. Face and non-face images have been used to generate HOG features, and a linear SVM is trained using these features. Using a sliding window approach, each patch of the image is classified as being face or non-face. This project is based on the paper by Dalal and Triggs 2005 for the detection of faces in images using a sliding window. The project can be divided into the following steps.


Extractinng postive features using face images.

Extracting negative features using the non-face images.

Training a linear SVM using the positive and negative features.

Using a sliding window to evaluate each patch of the image as being face or non-face.


1) To test the decision tree classifier please uncomment the functions decision_tree_classifier and run_detector tree in student_code.py

2) Please also uncomment the last few lines in the python notebook under the heading "Decision Tree"
