# A Deep Learning Approach For Melanoma Detection

### Motivation

Melanoma is a cancer of the neural crest–derived cells that provide pigmentation to skin and other tissues. Over the past 4 decades, the incidence of melanoma has increased more rapidly than that of any other malignancy in the United States<sup>i</sup>. Survival of patients with malignant melanoma is directly related to early detection<sup>ii</sup>.  The current techniques to identify a melanoma is a manual scan of the skin, but the only way to accurately diagnose melanoma is with a biopsy. There is a scope for developing more accurate, non-invasive, automated techniques to identify melanomas in order to reduce medical costs and facilitate early detection.

### Prior Work

Previous approaches are oriented toward pattern recognition. These previous approaches can be classified based on whether they target global features or local features. The global approaches follow the outline of lesion segmentation, feature extraction, feature selection, and lesion classification. The local approaches on the other hand divide the image into patches and extract features from these patches. All these approaches use Statistical Learning, Naive Bayes, SVMs etc.<sup>iii</sup> However, CNNs have proved to be very effective at tasks such as segmentation, localization and feature extraction and should definitely be explored for these tasks. A more recent approach used CNNs pre-trained on the ILSVRC 2012 data for feature extraction followed by SVMs for classification using the features extracted from the pre-trained CNN<sup>iv</sup>. This approach however, did not customize the CNN to the task at hand.

### Approach and Techniques

We propose to target 3 different tasks:

**Segmentation** : A good start would be trying out fully convolutional networks that have shown success with semantic segmentation<sup>v</sup>.

**Feature extraction** : We plan to use R-CNNs or window methods for similar reasons<sup>vi</sup>

**Classification** : We are planning to use CNNs followed by fully connected layers for the classification challenge. We will try different architectures that already exist and try to improve them by adapting them to the task at hand.

Above mentioned tasks are part of ISBI 2016 challenge, we plan to participate if we happen to meet challenge deadline (Apr 1)

### Datasets

In order to train our network, we are planning to use the following datasets:

- ISBI 2016 Challenge dataset: Training data (~900 images) + Test data (~350 images). This dataset is a subset of the ISIC Archive which is part of the "International Skin Imaging Collaboration: Melanoma Project"
- ImageNet from the Large Scale Visual Recognition Challenge: We plan to use this large dataset for pre-training.
- DermNet: DermNet is the largest independent photo dermatology source dedicated to online medical education through articles, photos and video. It is a public dataset that contains over 23,000 images of skin disease.

### Experimental Methodology

- The first step of our approach is to pre-train our model and get it to converge with our dataset.
- Once we get it to converge, various metrics will be used to test the network including using metrics Jaccard Index, Dice coefficient for segmentation, sensitivity, specificity and accuracy for the remaining tasks.
- Depending on the performance, we will continue to tweak our network by adding more components, varying hyperparameters, changing the architecture until we obtain satisfactory performance levels across all domains.
- We then plan to do an extensive analysis of the final network and results obtained.

### Group Tasking

Our project is divided into three main components, building the network, tweaking the network and analysis of the network. Though the tasks are divided amongst the team members; we plan to collectively involve all members at every step to ensure everyone understands and learns all aspects.

Task division is described below,

- Building (Mohit): Pre-training and basic implementation of network
- Tweaking (Nandita): Playing with hyperparameters and network architecture
- Analysis (Sahbi): Performance evaluations and insights from data

#References
<sup>i</sup>
Chudnovsky, Yakov, Paul A. Khavari, and Amy E. Adams. "Melanoma genetics and the development of rational therapeutics." The Journal of clinical investigation 115.4 (2005): 813-824.

<sup>ii</sup>
 Rigel, Darrell S., and John A. Carucci. "Malignant melanoma: prevention, early detection, and treatment in the 21st century." _CA: a cancer journal for clinicians_ 50.4 (2000): 215-236.

<sup>iii</sup>
 Barata, Catarina, et al. "Two systems for the detection of melanomas in dermoscopy images using texture and color features." _Systems Journal, IEEE_ 8.3 (2014): 965-979.

<sup>iv</sup>
 Codella, Noel, et al. "Deep Learning, Sparse Coding, and SVM for Melanoma Recognition in Dermoscopy Images." _Machine Learning in Medical Imaging_. Springer International Publishing, 2015. 118-126.

<sup>v</sup>
 Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_. 2015.

<sup>vi</sup>
 Girshick, Ross, et al. "Rich feature hierarchies for accurate object detection and semantic segmentation." _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2014.
