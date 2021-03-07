

import argparse
import csv
import logging
import multiprocessing
import os
import pickle
import pprint

from ruffus.proxy_logger import *
from scipy.stats import kendalltau, pearsonr, spearmanr

from predict_regulons import *


def parseCommandLineArguments():
    parser = argparse.ArgumentParser(prog="predict_regulons.py",
                                     description="""Performs all related operations to predict regulons
    """)
    # Mandatory arguments
    parser.add_argument("--probe_set_info","-pb",help="Please enter the name of the csv file containing all information about probe sets, genes and regulons",required=True)
    parser.add_argument("--output","-o",help="Please enter the name of the output directory. Download will be skipped if file is present",required=True)
    parser.add_argument("--gtf","-gtf",help = "Enter the GTF file",required=True)
    parser.add_argument("--genome","-g",help="Enter the genome fasta file",required=True)
    parser.add_argument("--counts","-c",help="Enter the name of the counts file",required=True)
    parser.add_argument("--star_index","-star_index",help="Enter the location of STAR index",required = True)
    parser.add_argument("--transcript_to_gene_map","-map",help="Enter the transcript to gene map",required = True)
    parser.add_argument("--genes_in_microarray","-gm",help = "Genes represented in microarrat",required=True)
    
    
    # Optional arguments
    parser.add_argument("--cpu","-n",help="Enter the number of CPUs to be used.",default=1)
    return parser.parse_args()

def calculateSpearmanInParallel(eachinput):
    output=[]
    for row in eachinput:
        X,Y,gene1,gene2=row
        output.append([spearmanr(X,Y)[0],gene1,gene2])
    return output

def calculatePearsonrInParallel(eachinput):
    output=[]
    for row in eachinput:
        X,Y,gene1,gene2=row
        output.append([pearsonr(X,Y)[0],gene1,gene2])
    return output

def calculateKendallTauInParallel(eachinput):
    output=[]
    for row in eachinput:
        X,Y,gene1,gene2=row
        output.append([kendalltau(X,Y)[0],gene1,gene2])
    return output

def computeGeneCorrelation(options,logging_mutex,logger_proxy,countsdata_matrix,n_comp):
    countsdata_matrix_n_comp = {}
    for gene in countsdata_matrix:
        countsdata_matrix_n_comp[gene] = countsdata_matrix[gene][:n_comp]
    
    pearson_filename = f"{options.output}/pearson_{n_comp}.txt"
    spearman_filename = f"{options.output}/spearman_{n_comp}.txt"
    kendalltau_filename = f"{options.output}/kendall_{n_comp}.txt"
    genes_to_be_skipped_pr=[]
    genes_to_be_skipped_sp=[]
    genes_to_be_skipped_kt=[]
    if os.path.exists(pearson_filename)==False:
        fhw_pearson_filename = open(pearson_filename,"w",buffering = 1)
    else:
        genes_to_be_skipped_pr=[]
        fhr=open(pearson_filename,"r")
        for line in fhr:
            gene1,gene2,corr=line.strip().split("\t")
            genes_to_be_skipped_pr.append(gene1)
        fhr.close()
        genes_to_be_skipped_pr=list(set(genes_to_be_skipped_pr))
        fhw_pearson_filename = open(pearson_filename,"a",buffering = 1)
        
    if os.path.exists(pearson_filename)==False:
        fhw_spearman_filename = open(spearman_filename,"w",buffering = 1)
    else:
        genes_to_be_skipped_sp=[]
        fhr=open(pearson_filename,"r")
        for line in fhr:
            gene1,gene2,corr=line.strip().split("\t")
            genes_to_be_skipped_sp.append(gene1)
        fhr.close()
        genes_to_be_skipped_sp=list(set(genes_to_be_skipped_sp))
        fhw_spearman_filename = open(spearman_filename,"a",buffering = 1)
        
    if os.path.exists(pearson_filename)==False:
        fhw_kendalltau_filename = open(kendalltau_filename,"w",buffering = 1)
    else:
        genes_to_be_skipped_kt=[]
        fhr=open(pearson_filename,"r")
        for line in fhr:
            gene1,gene2,corr=line.strip().split("\t")
            genes_to_be_skipped_kt.append(gene1)
        fhr.close()
        genes_to_be_skipped_kt=list(set(genes_to_be_skipped_kt))
        fhw_kendalltau_filename = open(kendalltau_filename,"a",buffering = 1)
    
    pool = multiprocessing.Pool(processes=int(options.cpu))    
    allinputs_sp,allinputs_pr,allinputs_kt = [],[],[]
    """genes_to_be_skipped=[]
    genes_to_be_skipped.extend(genes_to_be_skipped_sp)
    genes_to_be_skipped.extend(genes_to_be_skipped_pr)
    genes_to_be_skipped.extend(genes_to_be_skipped_kt)
    genes_to_be_skipped=set(genes_to_be_skipped)"""
    with logging_mutex:
        logger_proxy.info(f"Number of genes to be skipped Pearson {len(genes_to_be_skipped_pr)}")
        logger_proxy.info(f"Number of genes to be skipped Spearman {len(genes_to_be_skipped_sp)}")
        logger_proxy.info(f"Number of genes to be skipped Kendall-tau {len(genes_to_be_skipped_kt)}")
        print(f"Number of genes to be skipped Pearson {len(genes_to_be_skipped_pr)}")
        print(f"Number of genes to be skipped Spearman {len(genes_to_be_skipped_sp)}")
        print(f"Number of genes to be skipped Kendall-tau {len(genes_to_be_skipped_kt)}")
        
        sys.stdout.flush()
    
    genenames = list(countsdata_matrix.keys())
    for gene_num1 in range(len(genenames)):
        allinputs_per_gene_pr,allinputs_per_gene_sp,allinputs_per_gene_kt=[],[],[]
        gene1 = genenames[gene_num1]
        if len(set(countsdata_matrix_n_comp[gene1]))==1 and countsdata_matrix_n_comp[gene1][0]==0:
            continue
        
        if genenames[gene_num1] not in set(genes_to_be_skipped_pr):
            gene_num2=gene_num1+1
            while gene_num2!=len(genenames):
                gene2=genenames[gene_num2]
                allinputs_per_gene_pr.append([countsdata_matrix_n_comp[gene1],countsdata_matrix_n_comp[gene2],genenames[gene_num1],genenames[gene_num2]])
                gene_num2+=1
        
        if genenames[gene_num1] not in set(genes_to_be_skipped_sp):
            gene_num2=gene_num1+1
            while gene_num2!=len(genenames):
                gene2=genenames[gene_num2]
                allinputs_per_gene_sp.append([countsdata_matrix_n_comp[gene1],countsdata_matrix_n_comp[gene2],genenames[gene_num1],genenames[gene_num2]])
                gene_num2+=1
            
        if genenames[gene_num1] not in set(genes_to_be_skipped_kt):
            gene_num2=gene_num1+1
            while gene_num2!=len(genenames):
                gene2=genenames[gene_num2]
                allinputs_per_gene_kt.append([countsdata_matrix_n_comp[gene1],countsdata_matrix_n_comp[gene2],genenames[gene_num1],genenames[gene_num2]])
                gene_num2+=1
        
        with logging_mutex:
            logger_proxy.info(f"Processed {gene_num1} {len(allinputs_per_gene_pr)} {len(allinputs_per_gene_sp)} {len(allinputs_per_gene_kt)}")
        
        allinputs_pr.append(allinputs_per_gene_pr)
        allinputs_sp.append(allinputs_per_gene_sp)
        allinputs_kt.append(allinputs_per_gene_kt)
        
        #if len(allinputs)>=int(options.cpu):
        
        with logging_mutex:
            logger_proxy.info(f"Starting calculations with {n_comp} components and {gene_num1} gene")
        results=pool.map(calculatePearsonrInParallel,allinputs_pr)
        for row in results:
            for corr,gene1,gene2 in row:
                print(f"Writing to Pearson file {n_comp}")
                sys.stdout.flush()
                fhw_pearson_filename.write("\t".join([gene1,gene2,str(corr)])+"\n")
        
        results=pool.map(calculateSpearmanInParallel,allinputs_sp)
        for row in results:
            for corr,gene1,gene2 in row:
                print(f"Writing to Spearman file {n_comp}")
                sys.stdout.flush()
                fhw_spearman_filename.write("\t".join([gene1,gene2,str(corr)])+"\n")
        
        results=pool.map(calculateKendallTauInParallel,allinputs_kt)
        for row in results:
            for corr,gene1,gene2 in row:
                print(f"Writing to Kendall-tau file {n_comp}")
                sys.stdout.flush()
                fhw_kendalltau_filename.write("\t".join([gene1,gene2,str(corr)])+"\n")
        allinputs_sp,allinputs_pr,allinputs_kt = [],[],[]
        
    fhw_spearman_filename.close()
    fhw_pearson_filename.close()
    fhw_kendalltau_filename.close()
    pool.close()
    pool.join()

def configureLogger(options):
    os.system("rm -f "+options.output+"/calculate_correlations_progress.log")
    
    arguments={}
    arguments["file_name"]=options.output+"/calculate_correlations_progress.log"
    arguments["formatter"] = "%(asctime)s - %(name)s - %(levelname)6s - %(message)s"
    arguments["level"]     = logging.DEBUG
    arguments["delay"]     = False
    
    (logger_proxy,logging_mutex) = make_shared_logger_and_proxy (setup_std_shared_logger,"calculate_correlations", arguments)
    
    return logger_proxy,logging_mutex


def main():
    commandLineArg=sys.argv
    if len(commandLineArg)==1:
        print("Please use the --help option to get usage information")
    options=parseCommandLineArguments()
    
    os.system("mkdir -p "+options.output)
    logger_proxy,logging_mutex=configureLogger(options)
    readFromProbeSetFile(options)
    
    fhr = open(options.output+"/genes_to_regulons.tsv","r")
    gene_to_regulon = {}
    regulon_to_labels = {}
    labels_to_regulons = {}
    label_number = 1
    for line in fhr:
        gene,regulon = line.split("\t")
        if regulon=='X':
            gene_to_regulon[gene] = 0
            regulon_to_labels[regulon] = 0
        else:
            if regulon not in regulon_to_labels:
                regulon_to_labels[regulon] = label_number
                labels_to_regulons[label_number] = regulon
                gene_to_regulon[gene] = regulon_to_labels[regulon]
                label_number+=1
            else:
                gene_to_regulon[gene] = regulon_to_labels[regulon]
    fhr.close()
    
    options.pca_pkl_file = options.output+"/pca.pkl"
    #os.system("rm "+options.pca_pkl_file)
    if os.path.exists(options.pca_pkl_file)==False:
        countsdata = readCountsFile(options.counts,list(gene_to_regulon.keys()))
        countsdata_matrix_pca_all_components = performPCA(countsdata)
        pickle.dump(countsdata_matrix_pca_all_components,open(options.pca_pkl_file,"wb"))
    else:
        countsdata = readCountsFile(options.counts,list(gene_to_regulon.keys()))
        pca_data = pickle.load(open(options.pca_pkl_file,"rb"))
        countsdata_matrix_pca_all_components = {}
        for gene_num,gene in enumerate(countsdata):
            countsdata_matrix_pca_all_components[gene.strip()] = list(pca_data[gene_num])
    
    """for gene in countsdata_matrix_pca_all_components:
        print(gene,countsdata_matrix_pca_all_components[gene])
    return"""
    for pca_comp in list(range(150,501,25))[1:]:
        with logging_mutex:
            logger_proxy.info(f"Calling countsdata_matrix_pca_all_components with {pca_comp} components")
            print(f"Calling countsdata_matrix_pca_all_components with {pca_comp} components")
            sys.stdout.flush()
        computeGeneCorrelation(options, logging_mutex,logger_proxy,countsdata_matrix = countsdata_matrix_pca_all_components,n_comp=pca_comp)
    

if __name__ == "__main__":
    main()
