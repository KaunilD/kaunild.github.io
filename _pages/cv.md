---
layout: archive
title: "CV"
permalink: /cv
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

[Resume](https://kaunild.github.io/files/resume.pdf)

Education
======
* M.S. in CS, University of Colorado Boulder, 2020 (expected)
* B.E. in CS, University of Mumbai, 2013 - 2017.

Work experience
======
* __Computer Vision Engineer - Insylo Tech SLU (contract):__ _March 2018 - June 2018_
  * Developed Computer Vision Pipelines for Volumetric Estimation of Silos using 2D and 3D images.
  * CV Techniques such as Depth from Focus and depth from Laser Mesh Projection were implemented.
  * Researched the performance of GANs for generation of depth maps using 2D (monocular) images.  
  * Assessing the feasibility of depth sensing cameras (Astra Pro, Intel Realsense) in a production setting using Android device and Raspberry Pi.

* __Software Engineer in Machine Learning: Facebook:__ _September 2016 - December 2017_
  * Worked with Connectivity Labs and mentored by Prof. Ramesh Raskar.
  * Researched and implemented a pipeline for Visualization of Learned Features of a CNN based on SGD to improve model training for SegNet, VGGBn and UResNet architectures.
  * Created Data Annotation tools using Qt5 used by a team (5) of GIS Analysts.
  * Optimized rendering of 2D vector geometries on an Android App. This optimization made it possible for the Android Application to be used by low end phones and reduce costs by a factor of 10.
  * All the codes are open sourced to [facebookresearch/street-address](https://github.com/facebookresearch/street-addresses)

* __Software Engineer Intern: ISENSES INC.__ _January 2016 - January 2017_
  * Developed a Machine Learning pipeline for Disguised Face Detection.
  * Implemented a SegNet based feature detector to identify facial action units which were then used to classify disguised faces using an SVM Classifier.
  * Entire pipeline was optimized and implemented on a an FPGA and materialized into a product.

Publications
======
  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Awards
======
* __Best Paper Award__ for the research paper _“Robocodes: Towards Generative Street Addresses from Satellite Imagery”_ in __CVPR 2017 workshop on Earthvision.__
* __MS Imagine Cup 2017 Korea Semi-Finalist__ for our project: _COBRIX_  
