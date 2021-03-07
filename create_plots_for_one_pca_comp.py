import pprint
import sys
import argparse
import random
import os
from scipy.stats.stats import pearsonr
import itertools
from itertools import count
from numpy.random import sample
import numpy as np
import numpy.random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster.tests.test_k_means import n_samples
from scipy.constants.constants import alpha
from sklearn.metrics.ranking import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_val_predict
from sklearn import svm
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from unicodedata import decomposition
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_validate
#from sklearn.metrics import plot_precision_recall_curve
import glob
#sys.exit()


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
"""
LOAD THE MODULES

module purge
module load py-pandas/0.23.4-py3-sx6iffy
module load py-matplotlib/2.2.2-py3-ko25og4
module load py-numpy/1.15.2-py3-wwyx7ek 
module load py-pytest/3.7.2-py3-l65jw2j 
module load py-scipy/1.1.0-py3-zfiwiow
"""
def parseCommandLineArguments():
    """
    Parses the arguments provided through command line.
    Launch python machine_learning_classifiers_with_n_comp.py --help for more details
    """
    parser = argparse.ArgumentParser(prog="correlation.py",description="")
    
    parser.add_argument("-tr","--tr",help="Enter the name of the training data file",required=True)#d is the data file, can be microarray or RNA-seq data
    parser.add_argument("-te","--te",help="Enter the name of the testing data file",required=True)
    parser.add_argument("-g","--g",help="Enter the name of the gene list", required=True)#g is the genes belonging to a regulon
    parser.add_argument("-r","--regulon_name",help="Enter the name of the regulon",required=True) 
    #parser.add_argument("-p","--prefix",help="Enter the prefix of the pickle file you want to save for this model")
    parser.add_argument("-o","--output",help="Enter the name for the output directory", required=True)
    parser.add_argument("-n","--ncomp", help="Enter the number of pca components", required=True)
    parser.add_argument("-ft","--feature_type",help="Enter the feature type rc or mcc",required=True)
    parser.add_argument("-ds","--data_split",help="Enter how the training and testing data was split rand or cath",required=True)
    parser.add_argument("-dsn","--dataset_number",help="Enter the dataset number",required=True)
    return parser.parse_args()
    
def main():
    commandLineArg=sys.argv
    if len(commandLineArg)==1:
        print("Please use the --help option to get usage information")
    options = parseCommandLineArguments()
    #os.makedirs(options.output+"_dir", exist_ok = True)
    os.system("mkdir -p "+options.output)
    options.pickle_file_directory=options.output+"/pickle_files"
    os.system("mkdir -p "+options.pickle_file_directory)
    
    training_set, testing_set, geneArray = sanityCheck(options)
    X_training, Y_training = createTrainingDatasets(training_set, geneArray) # it is a better idea to construct the labels and the training data within the same function
    X_testing, Y_testing = createTrainingDatasets(testing_set, geneArray)
    scaler=preprocessing.StandardScaler().fit(X_training)
    X_train_normalized = scaler.transform(X_training)
    X_test_normalized = scaler.transform(X_testing)
    # Generating dummy data for testing purposes
    #X_train_normalized = X_train_normalized[:1000]
    #Y_training = [random.randint(1,1000) % 2 for _ in range(1000)]
    
    i = options.ncomp
    complete_info={}
    performTrainingAndTesting(X_train_normalized, X_test_normalized, Y_training, Y_testing, options, str(i))
    """for machine_learner in ["svm","sgd","rf","mlp","gbt"]:
        machine_learner,precision,recall,f1_test_score,matthews_corr,ROC_score,Y_pred_0,Y_pred_1,CV_clf_proba =performTrainingAndTesting(X_train_normalized, X_test_normalized, Y_training, Y_testing, options, str(i),machine_learner)
        complete_info[machine_learner]=[round(x,2) for x in [precision,recall,f1_test_score,matthews_corr,ROC_score,Y_pred_1]]"""
    """for machine_learner in ["mlp","gbt"]:
        machine_learner,precision,recall,f1_test_score,matthews_corr,ROC_score,Y_pred_0,Y_pred_1,CV_clf_proba =performTrainingAndTesting(X_train_normalized, X_test_normalized, Y_training, Y_testing, options, str(i),machine_learner)
        precision, recall, thresholds = precision_recall_curve(Y_testing, CV_clf_proba[:,1])
        plot_precison_recall_curve(precision, recall, machine_learner, options)"""
        
    #complete_info_pd=pd.DataFrame.from_dict(complete_info,orient='index')
    #complete_info_pd.columns = ["precision","recall","f1","mcc","auc","pos"]
    #pprint.pprint(complete_info_pd)
    """pickle_filename=glob.glob("/work/LAS/mash-lab/bhandary/analysis_regulon_prediction/pickle_files/rc*PCA20_Photosynthesis*pickle")  
    print (pickle_filename)                                                        
    for pickle_file in pickle_filename:
        colors = ["red", "blue", "green", "purple", "brown", "black"]
        i = 0
        machine_learner = ["gbt", "mlp", "rf", "sgd", "svm"]
        j = 0
        CV_clf=pickle.load(open(pickle_file,"rb"))
        CV_clf_proba = CV_clf.predict_proba(X_test_normalized)
        Y_pred = CV_clf.predict(X_test_normalized)
        precision = precision_score(Y_testing, Y_pred, average = 'binary')
        recall = recall_score(Y_testing, Y_pred, average = 'binary')
        f1_test_score = f1_score(Y_testing, Y_pred, average = 'binary')
        matthews_corr = matthews_corrcoef(Y_testing, Y_pred)
        ROC_score = roc_auc_score(Y_testing, Y_pred)
        #plot_precison_recall_curve_condensed(precision, recall, pca_n_comp, machine_learner)
        #plot_precision_recall_curve(X_test_normalized, Y_testing, ax = plt.gca(), color = colors[i], name = machine_learner[j])
        plt.plot(recall, precision, color = colors[i], name = machine_learner[j])
        i += 1
        j += 1
    plt.savefig('Precision-Recall Curve PCA_20')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve')
    plt.show()"""
    #sys.stdout.flush()
    
def sanityCheck(options):
    training_set = readDataFile(options.tr)
    testing_set = readDataFile(options.te)
    geneArray = readGeneList(options.g)
    return training_set, testing_set, geneArray

def readDataFile(datafile):
    countdata = open(datafile, 'r')
    countArray = {}
    for line in countdata:
        line = line.strip()
        line = line.split()
        if ";" in line[0]:
            for id in line[0].split(";"):
                countArray[id.upper()] = list(map(float,line[1:]))
        else:
            countArray[line[0].upper()] = list(map(float,line[1:]))
    return countArray
        
def readGeneList(genelist):    
    genes = open(genelist, 'r')
    geneArray = []
    for line in genes:
        line = line.strip()
        #print (line)
        if "," not in line:
            geneArray.append(line.upper())
        else:
            geneArray.extend([x.strip().upper() for x in line.split(",")])
    return geneArray

def randomSample(a):
    #sample_without_replacement()
    return np.random.choice(a, size= 500, replace = False, p = None)

def average(list):
    return sum(list) / len(list)

def pairwiseCorrelation(a,b):
    c = pearsonr(a, b)
    return c

def createTrainingDatasets(countArray, geneArray):
    Y_training = []
    X_training = []
    for  gene in countArray:
        X_training.append(np.array(countArray[gene]))
        if gene in geneArray:
            Y_training.append(1)
        else:
            Y_training.append(0)
    
    """for i in range(len(X_training)):
        print(X_training[i][:5],Y_training[i])"""
    
    return X_training, Y_training

def performTrainingAndTesting(X_train_normalized_pca, X_test_normalized_pca, Y_train, Y_test, options, pca_n_comp):
    #return machine_learner,1,2,3,4,5,6,7
    #pickle_filename=options.pickle_file_directory+"/"+options.regulon_name+"_"+machine_learner+"_pca_"+pca_n_comp+".pickle"
    #print ("Y_train_0=", Y_train.count(0), "Y_train_1=", Y_train.count(1), "Y_test_0=", Y_test.count(0), "Y_test_1=", Y_test.count(1))
    """pickle_filename=glob.glob(options.pickle_file_directory+"/"+"_".join([options.feature_type,
                                                                options.data_split,
                                                                ("DS"+options.dataset_number if "DS" not in options.dataset_number else options.dataset_number),
                                                                "PCA20",
                                                                options.regulon_name,
                                                                machine_learner])+".pickle")""" 
    PCA = ["PCA30", "PCA40", "PCA50", "PCA60", "PCA70", "PCA80", "PCA90", "PCA100", "PCA150", "PCA200", "PCA250", "PCA500"]  
    for comp in PCA:
        colors = ["red", "blue", "green", "purple", "brown", "black"]
        linestyle = ['-', '--', '-.', ':', '']
        i = 0
        machine_learner = ["gbt", "mlp", "rf", "sgd", "svm"] 
        while i<len(machine_learner):
            pickle_file = options.pickle_file_directory+"/"+"_".join([options.feature_type,"rand","DS0",comp,"Photosynthesis",machine_learner[i]])+".pickle"
            CV_clf=pickle.load(open(pickle_file,"rb"))
            CV_clf_proba = CV_clf.predict_proba(X_test_normalized_pca)
            Y_pred = CV_clf.predict(X_test_normalized_pca)
            precision = precision_score(Y_test, Y_pred, average = 'binary')
            recall = recall_score(Y_test, Y_pred, average = 'binary')
            f1_test_score = f1_score(Y_test, Y_pred, average = 'binary')
            matthews_corr = matthews_corrcoef(Y_test, Y_pred)
            ROC_score = roc_auc_score(Y_test, Y_pred)
            precision, recall, _ = precision_recall_curve(Y_test, CV_clf_proba[:,1])
            #plot_precison_recall_curve_condensed(precision, recall, pca_n_comp, machine_learner)
            #plot_precision_recall_curve(X_test_normalized_pca, Y_test, ax = plt.gca(), color = colors[i], name = machine_learner[j])
            print("Recall",recall)
            print("Precision",precision)
            sys.stdout.flush()
            plt.plot(recall, precision, color = colors[i], label = machine_learner[i])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="lower left")
            i+=1
        os.system('rm Precision_recall_curve_'+str(comp)+'_Photosynthesis.png')
        plt.title('Precision_recall_curve_'+str(comp)+'_Photosynthesis')
        plt.savefig('Precision_recall_curve_'+str(comp)+'_Photosynthesis.png', dpi=1200)
        plt.show()
        plt.close()

    #return machine_learner,precision,recall,f1_test_score,matthews_corr,ROC_score,list(Y_pred).count(0),list(Y_pred).count(1), CV_clf_proba

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('ROC_ML.png')
    plt.show()

"""def plot_precison_recall_curve(precision, recall, machine_learner, options):
    print(recall)
    print(precision)
    plt.plot(recall, precision, color='orange')
    #plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve '+options.feature_type+"_"+options.data_split+"_"+options.dataset_number+"_"+options.ncomp+"_"+options.regulon_name+"_"+machine_learner)
    plt.legend()
    plt.savefig('Precision_Recall_ML_'+options.feature_type+"_"+options.data_split+"_"+options.dataset_number+"_"+options.ncomp+"_"+options.regulon_name+"_"+machine_learner+'.png')
    plt.show()
    plt.close()"""

def plot_precison_recall_curve_condensed(precision, recall, pca_n_comp, machine_learner):
    plt.plot(recall, precision, color='orange')
    #plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    

    
if __name__ == "__main__":
    main()