import os, sys
import glob

classifier = ["sgd", "svm", "rf", "gbt", "mlp"]
filename = "/work/LAS/mash-lab/bhandary/analysis_regulon_prediction/pickle_files/rc_rand_DS0_PCA20_Photosynthesis_*"

for file in filename:
    