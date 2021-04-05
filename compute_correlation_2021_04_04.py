import math
import argparse
import csv
import logging
import multiprocessing
import os
import pickle
import pprint
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from predict_regulons import *

"""
ROOT_DIRECTORY = /project/....
CPU = 30

python compute_correlation_2021_04_04.py \
--correlation_filename \
--train_test_gene_file \
--outputfilename \
--cpu $CPU\
1> 
2> 

"""

def parseCommandLineArguments():
    parser = argparse.ArgumentParser(prog="compute_correlation_2021_04_04.py",
                                     description="""""")
    # Mandatory arguments
    parser.add_argument("--correlation_filename", "-c", help = "Enter the name of the counts file in csv format", required = True)
    parser.add_argument("--train_test_gene_file", "-ttg", help = "Enter the filename that has the list of training and testing genes", required = True)
    parser.add_argument("--outputfilename", "-o", help = "Please enter the name of the output filename", required = True)
    
    # Optional arguments
    parser.add_argument("--cpu","-t",help="Enter the number of CPUs to be used.",default=1)
    
    return parser.parse_args()
    
def readTrainAndTestGenes(options):
    train_pos, train_neg, test_pos, test_neg = [], [], [], []
    fhr = open(options.train_test_gene_file, "r")
    for line in fhr:
        col1,label,list_of_genes = line.strip().split("\t")
        list_of_genes = [g.strip() for g in list_of_genes.split(",")]
        genes_to_be_removed = []
        for g in list_of_genes:
            if g!=g.strip():
                genes_to_be_removed.append(g)
        print(genes_to_be_removed)
        for g in genes_to_be_removed:
            list_of_genes.remove(g)
        if col1 == "Train":
            if label == "1":
                train_pos.extend(list_of_genes)
            elif label == "0" or label == "2":
                train_neg.extend(list_of_genes)
        elif col1 == "Test":
            if label == "1":
                test_pos.extend(list_of_genes)
            elif label == "0" or label == "2":
                test_neg.extend(list_of_genes)
    fhr.close()
    
    return train_pos, train_neg, test_pos, test_neg
    
def readCorrelationFile(options):
    correlations = {}
    fhr = open(options.correlation_filename, "r")
    for line in fhr:
        gene1,gene2,counts= line.strip().split()
        if gene1 not in correlations:
            correlations[gene1] = {}
        if gene2 not in correlations:
            correlations[gene2] = {}
        correlations[gene1][gene2] = np.arctanh(float(counts))
        correlations[gene2][gene1] = np.arctanh(float(counts))
    fhr.close()
    return correlations

def calculateAverageCorrelationOfOneGeneWithAGroupOfGenes(gene,gene_list,correlations):
    average = 0
    for g in gene_list:
        average += correlations[gene][g]
    return average/len(gene_list)

def calculateCorrelationForEachGene(train_pos, train_neg, test_pos, test_neg,correlations,options):
    fhw=open(options.outputfilename,"w")
    all_test_genes = []
    all_test_genes.extend(test_pos)
    all_test_genes.extend(test_neg)
    for gi in all_test_genes:
        z_i_plus = calculateAverageCorrelationOfOneGeneWithAGroupOfGenes(gi,train_pos,correlations)
        z_i_minus = calculateAverageCorrelationOfOneGeneWithAGroupOfGenes(gi,train_neg,correlations)
        s_plus = 0
        for gj in train_pos:
            s_plus += (correlations[gi][gj] - z_i_plus)**2
        s_plus = s_plus/(len(train_pos)-1)
        
        s_minus = 0
        for gj in train_neg:
            s_minus += (correlations[gi][gj] - z_i_minus)**2
        s_minus = s_minus/(len(train_neg)-1)
        
        s_p = math.sqrt(((len(train_pos)-1)*s_plus**2 + (len(train_neg)-1)*s_minus**2)/(len(train_pos)+len(train_neg)-2))
        d_corr_gi = (z_i_plus-z_i_minus)/s_p*math.sqrt(1/len(train_pos)+1/len(train_neg))
        fhw.write(f"{gi}\t{d_corr_gi}\n")
        
    

def main():
    commandLineArg=sys.argv
    if len(commandLineArg)==1:
        print("Please use the --help option to get usage information")
    options=parseCommandLineArguments()
    
    train_pos, train_neg, test_pos, test_neg = readTrainAndTestGenes(options)
    
    correlations = readCorrelationFile(options)
    
    calculateCorrelationForEachGene(train_pos, train_neg, test_pos, test_neg,correlations,options)
    
if __name__ == "__main__":
    main()
    
    