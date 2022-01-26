
<h1> Malonylation-site-prediction </h1>

 Lysine malonylation is one of the most important post-translational modifications (PTMs). It affects the functionality of cells. Malonylation site prediction in proteins can unfold the mechanisms of cellular functionalities. Experimental methods are one of the due prediction approaches. But they are typically costly and time-consuming to implement. Recently, methods based on machine-learning solutions have been proposed to tackle this problem. Such practices have been shown to reduce costs and time complexities and increase accuracy. However, these approaches also have specific shortcomings, including inappropriate feature extraction out of protein sequences, high-dimensional features, and inefficient underlying classifiers. A machine learning-based method is proposed in this paper to cope with these problems. In the proposed approach, seven different features are extracted. Then, the extracted features are combined, ranked based on the Fisher’s score (F-score), and the most efficient ones are selected. Afterward, malonylation sites are predicted using various classifiers. Simulation results show that the proposed method has acceptable performance compared with some state-of-the-art approaches. In addition, the  XGBOOST classifier, founded on extracted features such as TFCRF, has a higher prediction rate than the other methods.
 
 

 <h1> Requirement: </h1> 

This program is suitable for Python3 <br>

This program uses the FScore method in the feature selection phase. <br>

In the classifier folder, we have implemented various classifier methods. <br>
In the data folder, There is original data. <br>
In the classifier folder, we have implemented various classifier methods. <br>
In the Feature_extraction folder, we have implemented various feature extraction methods. Note that the <b> weight folder </b> is the same as the <b> TFCRF </b> method.<br>



<h1> Malonylation-site-prediction: </h1>
For this purpose it is only need to run main.py file. <br>


<h1> Feedback: </h1>
Its pleasure for me to have your comment. j.pirgazi@mazust.ac.ir

