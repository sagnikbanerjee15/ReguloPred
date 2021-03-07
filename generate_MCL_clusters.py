import argparse
import copy
import logging
import multiprocessing
import os
import sys
import time

from ruffus.proxy_logger import *
from  scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import pprint


def parseCommandLineArguments():
    parser = argparse.ArgumentParser(prog="generate_MCL_clusters.py",description="")
    
    parser.add_argument("--output_directory","-o",help="Enter the name of the directory where all other operations will be performed",required=True)
    parser.add_argument("--gene_counts","-d",help="Enter the file containing all the counts for each gene",required=True)
    #parser.add_argument("--gene_group_filenames","-g",help="Enter a list of filenames each containing genes with the same function. Please note that you can provide multiple gene groups.",nargs="+",required=True)
    #parser.add_argument("--gene_group_names","-n",help="Enter the name of each of the gene groups",nargs="+",required=True)
    parser.add_argument("--cpu","-p",help="Enter the number of CPU cores to be used",default=1)
    parser.add_argument("--version","-v",help="",default="version 1.1d")
    #parser.add_argument("--list_of_genes","-l",help="Enter the file containing the list of newline separated genes",required=True)
    parser.add_argument("--pca","-pca",help="Enter the number of pca components",required=True)
    
    # Advanced Arguments
    parser.add_argument("--theshold_adjacency_matrix","-adj_thresh",default=[0.5],help="",nargs='+')
    parser.add_argument("--force","-f",help="Forces the pipeline to run all the steps. The pipeline, by default, will skip steps that has already been executed.",action="store_true")
    parser.add_argument("--verbose","-verb",help="Verbosity levels are 1 through 3. Values greater than 3 are interpreted as 3",default=1)
    
    # Suppressed Arguments
    parser.add_argument("--whole_data_pd",help=argparse.SUPPRESS)
    parser.add_argument("--num_samples",help=argparse.SUPPRESS)
    parser.add_argument("--spearman_correlation_filename",help=argparse.SUPPRESS)
    parser.add_argument("--pearson_correlation_filename",help=argparse.SUPPRESS)
    parser.add_argument("--kendalltau_correlation_filename",help=argparse.SUPPRESS)
    parser.add_argument("--skip_spearman",help=argparse.SUPPRESS)
    parser.add_argument("--skip_pearson",help=argparse.SUPPRESS)
    parser.add_argument("--skip_kendalltau",help=argparse.SUPPRESS)
    
    return parser.parse_args()

def validateCommandLineArguments(options,logger_proxy,logging_mutex):
    """if len(options.gene_group_filenames)!=len(options.gene_group_names):
        with logging_mutex:
            logger_proxy.info("Number of gene groups and names of gene groups are not equal")
        sys.exit()
    flag=0
    for filename in options.gene_group_filenames:
        if os.path.exists(filename)==False:
            with logging_mutex:
                logger_proxy.info(filename+" does not exist")
            flag=1
    if flag==1:
        sys.exit()"""
    
    options.kendalltau_correlation_filename=options.output_directory+"/kendall_"+options.pca+".txt"
    options.spearman_correlation_filename=options.output_directory+"/spearman_"+options.pca+".txt"
    options.pearson_correlation_filename=options.output_directory+"/pearson_"+options.pca+".txt"
    if options.force==True:
        options.skip_spearman=False
        options.skip_pearson=False
        options.skip_kendalltau=False
    else:
        options.skip_spearman=False
        options.skip_pearson=False
        options.skip_kendalltau=False
        """if os.path.exists(options.kendalltau_correlation_filename)==True:
            options.skip_kendalltau=True
        if os.path.exists(options.spearman_correlation_filename)==True:
            options.skip_spearman=True
        if os.path.exists(options.pearson_correlation_filename)==True:
            options.skip_kendalltau=True"""
        pass
    options.verbose=3 if int(options.verbose)>=3 else int(options.verbose)
    
def configureLogger(options):
    os.system("mkdir -p "+options.output_directory)
    os.system("rm "+options.output_directory+"/progress.log")
    
    arguments={}
    arguments["file_name"]=options.output_directory+"/progress.log"
    arguments["formatter"] = "%(asctime)s - %(name)s - %(levelname)6s - %(message)s"
    arguments["level"]     = logging.DEBUG
    arguments["delay"]     = False
    
    (logger_proxy,logging_mutex) = make_shared_logger_and_proxy (setup_std_shared_logger,"ORF_FUNC", arguments)
    
    return logger_proxy,logging_mutex

def readCountData(filename):
    fhr = open(filename, 'r')
    sample_names = fhr.readline().strip().split()[1:]
    count_array = {}
    for line in fhr:
        line = line.strip()
        line = line.split()
        if ";" in line[0]:
            for id in line[0].split(";"):
                count_array[id.upper()] = list(map(float,line[1:]))
        else:
            count_array[line[0].upper()] = list(map(float,line[1:]))
    pprint.pprint(count_array)
    return count_array,sample_names

def readGeneGroupInformation(gene_counts,gene_group_filenames):
    for filename in gene_group_filenames:
        fhr = open(filename, 'r')
        gene_array = []
        for line in fhr:
            line = line.strip()
            #print (line)
            if "," not in line:
                gene_array.append(line.upper())
            else:
                gene_array.extend([x.strip().upper() for x in line.split(",")])
        fhr.close()
        for gene in gene_counts:
            if gene in set(gene_array):
                gene_counts[gene].append(1)
            else:
                gene_counts[gene].append(0)
    return gene_counts

def readInputData(options,logger_proxy,logging_mutex):
    # Create a dictionary of all the gene counts
    gene_counts,sample_names=readCountData(options.gene_counts)
    with logging_mutex:
        logger_proxy.info("Reading of gene counts completed: readCountData(options.gene_counts) execution successful")
    
    # Create an empty Pandas dataframe
    column_names=[]
    column_names.extend(sample_names)
    #column_names.extend(options.gene_group_names)
    options.num_samples=len(sample_names)
    """
    # Read gene information
    gene_counts=readGeneGroupInformation(gene_counts,options.gene_group_filenames)
    with logging_mutex:
        logger_proxy.info("Reading of gene groups completed: readGeneGroupInformation(gene_counts,options.gene_group_filenames) execution successful")
    """
    # Update the pandas dataframe with rows from the dictionary
    whole_data_pd=pd.DataFrame.from_dict(gene_counts,orient='index',columns=column_names)
    with logging_mutex:
        logger_proxy.info("Number of RNA-Seq samples "+str(len(sample_names)))
        #logger_proxy.info("Number of gene groups to be probed "+str(len(options.gene_group_filenames)))
    
    options.whole_data_pd=whole_data_pd


def runMCL(eachinput):
    # Create mcl inputfile from raw correlation file using the provided threshold
    options,logger_proxy,logging_mutex,threshold,raw_correlation_file,filename_for_mcl_input,filename_for_mcl_output_prefix,correlation_type=eachinput
    if os.path.exists(filename_for_mcl_input)==False:
        fhr=open(raw_correlation_file,"r")
        fhw=open(filename_for_mcl_input,"w")
        fhw.write("---8<------8<------8<------8<------8<---\n")
        for line in fhr:
            gene1,gene2,correlation=line.strip().split("\t")
            if float(correlation)>threshold:
                fhw.write("\t".join([gene1,gene2,"1"])+"\n")
            else:
                fhw.write("\t".join([gene1,gene2,"0"])+"\n")
        fhw.write("--->8------>8------>8------>8------>8---\n")
        fhw.close()
        fhr.close()
        with logging_mutex:
            logger_proxy.info("MCL input file created for "+correlation_type+" and threshold "+str(threshold))
    else:
        with logging_mutex:
            logger_proxy.info("MCL input file exists for "+correlation_type+" and threshold "+str(threshold))
    

    cmd="mcl "+filename_for_mcl_input+" --abc "
    cmd+=" -te "+str(options.cpu)
    cmd+=" -o "+filename_for_mcl_output_prefix+".cluster"
    if os.path.exists(filename_for_mcl_output_prefix+".cluster")==False:
        os.system(cmd)
    with logging_mutex:
        logger_proxy.info("MCL clusters generated for "+correlation_type+" and threshold "+str(threshold))

def performMarkovClustering(options,logger_proxy,logging_mutex):
    if options.skip_spearman==False:
        raw_correlation_file=options.spearman_correlation_filename
        for threshold in options.theshold_adjacency_matrix:
            filename_for_mcl_input=options.output_directory+"/spearman_threshold_pca_"+str(options.pca)+"_"+str(threshold)+".correlation"
            filename_for_mcl_output_prefix=options.output_directory+"/spearman_threshold_"+str(options.pca)+"_"+str(threshold)
            runMCL([options,logger_proxy,logging_mutex,float(threshold),raw_correlation_file,filename_for_mcl_input,filename_for_mcl_output_prefix,"Spearman"])
    
    if options.skip_pearson==False:
        raw_correlation_file=options.pearson_correlation_filename
        for threshold in options.theshold_adjacency_matrix:
            filename_for_mcl_input=options.output_directory+"/pearson_threshold_"+str(options.pca)+"_"+str(threshold)+".correlation"
            filename_for_mcl_output_prefix=options.output_directory+"/pearson_threshold_"+str(options.pca)+"_"+str(threshold)
            runMCL([options,logger_proxy,logging_mutex,float(threshold),raw_correlation_file,filename_for_mcl_input,filename_for_mcl_output_prefix,"Pearson"])
    
    if options.skip_kendalltau==False:
        raw_correlation_file=options.kendalltau_correlation_filename
        for threshold in options.theshold_adjacency_matrix:
            filename_for_mcl_input=options.output_directory+"/kendalltau_threshold_"+str(options.pca)+"_"+str(threshold)+".correlation"
            filename_for_mcl_output_prefix=options.output_directory+"/kendalltau_threshold_"+str(options.pca)+"_"+str(threshold)
            runMCL([options,logger_proxy,logging_mutex,float(threshold),raw_correlation_file,filename_for_mcl_input,filename_for_mcl_output_prefix,"Kendalltau"])


def computeModularityForEachClusteringInParallel(eachinput):
    node_to_adjacent_node,cluster_filename,total_edges,outputfilename,logger_proxy,logging_mutex=eachinput
    fhr=open(cluster_filename,"r")
    clusters=[]
    for line in fhr:
        clusters.append(line.strip().split())
    fhr.close()
    
    # Calculating modularity
    m=0
    for cluster_num,cluster in enumerate(clusters):
        eii=0
        ai=0
        for gene1 in cluster:
            if gene1 in node_to_adjacent_node:
                eii+=len(set(node_to_adjacent_node[gene1]) & set(cluster))
                ai+=len(set(node_to_adjacent_node[gene1]) - set(cluster))
        ai+=eii
        m+=(eii/total_edges)-(ai*ai)/(total_edges*total_edges) 
    fhw=open(outputfilename,"w")
    fhw.write(str(m))
    fhw.close()
    with logging_mutex:
        logger_proxy.info("Modularity computation finished for "+outputfilename.split("/")[-1])


def computeModularityForEachClustering(options,logger_proxy,logging_mutex):
    #for raw_correlation_filename in [options.spearman_correlation_filename,options.pearson_correlation_filename,options.kendalltau_correlation_filename]:
    correlation_type=["spearman","pearson","kendalltau"]
    """if int(options.cpu)<=10:
        pool = multiprocessing.Pool(processes=int(options.cpu))
    else:
        pool = multiprocessing.Pool(processes=10)    
    allinputs=[]"""
    
    for correlation_num,raw_correlation_filename in enumerate([options.spearman_correlation_filename,options.pearson_correlation_filename,options.kendalltau_correlation_filename]):
        for threshold in options.theshold_adjacency_matrix:
            raw_correlation_filename=options.output_directory+"/"+correlation_type[correlation_num]+"_threshold_"+str(threshold)+".correlation"
            fhr=open(raw_correlation_filename,"r")
            node_to_adjacent_node={}
            total_edges=0
            prev_gene=""
            for line in fhr:
                if "---" in line:continue
                gene1,gene2,edge=line.strip().split()
                if edge=="0":continue
                if prev_gene=="" or prev_gene!=gene1:
                    prev_gene=gene1
                    node_to_adjacent_node[gene1]=[]
                node_to_adjacent_node[gene1].append(gene2)
                total_edges+=1  
            fhr.close()
            for inflation in [i / 10 for i in range(11, 51)]:
                cluster_filename=options.output_directory+"/"+correlation_type[correlation_num]+"_threshold_"+str(threshold)+"_inflation_"+str(inflation)+".cluster"
                outputfilename=options.output_directory+"/"+correlation_type[correlation_num]+"_threshold_"+str(threshold)+"_inflation_"+str(inflation)+".modularity"
                #allinputs.append([node_to_adjacent_node,cluster_filename,total_edges,outputfilename,logger_proxy,logging_mutex])
                computeModularityForEachClusteringInParallel([node_to_adjacent_node,cluster_filename,total_edges,outputfilename,logger_proxy,logging_mutex])


def selectBestClusters(options,logger_proxy,logging_mutex):
    correlation_type=["spearman","pearson","kendalltau"]
    """for correlation_num,raw_correlation_filename in enumerate([options.spearman_correlation_filename,options.pearson_correlation_filename,options.kendalltau_correlation_filename]):
        for threshold in options.theshold_adjacency_matrix:
            for inflation in [i / 10 for i in range(11, 51)]:
                print(correlation_type[correlation_num],threshold,inflation,open(options.output_directory+"/"+correlation_type[correlation_num]+"_threshold_"+str(threshold)+"_inflation_"+str(inflation)+".modularity","r").read())
    """
    # Read in the TPM values
    gene_counts=options.whole_data_pd.values.tolist()
    list_of_genes=[]
    fhr=open(options.list_of_genes,"r")
    """for line in fhr:
        if "Gene_ID" in line:continue
        list_of_genes.append(line.strip().split("\t")[0])"""
    for line in fhr:
        #print (line)
        """line = line.strip().split("\n")
        print (line)"""
        list_of_genes.append(line.strip("\n"))
        print (line)
    print (list_of_genes)
    fhr.close()
    
    for correlation_type in ["spearman","pearson","kendalltau"]:
        for threshold in options.theshold_adjacency_matrix:
            for inflation in [i / 10 for i in range(11, 51)]:
                cluster_filename=options.output_directory+"/"+correlation_type+"_threshold_"+str(threshold)+"_inflation_"+str(inflation)+".cluster"
                fhr=open(cluster_filename,"r")
                gene_clusters=[]
                for line in fhr:
                    gene_clusters.append(line.strip().split())
                fhr.close() 
                gene_to_cluster={}
                for cluster_num,cluster in enumerate(gene_clusters):
                    for gene in cluster:
                        gene_to_cluster[gene]=cluster_num
                pprint.pprint(gene_to_cluster)
                Y=[]
                for gene in list_of_genes:
                    Y.append(gene_to_cluster[gene])
                X=gene_counts
                print (len(X), len(Y))
                if len(gene_clusters)==1:
                    silhouette_score=""
                    davies_bouldin_score=""
                else:
                    silhouette_score=metrics.silhouette_score(X, Y)
                    davies_bouldin_score=metrics.davies_bouldin_score(X,Y)
                fhw=open(options.output_directory+"/"+correlation_type+"_threshold_"+str(threshold)+"_inflation_"+str(inflation)+".silhouette_score","w")
                fhw.write(str(silhouette_score)+"\n")
                fhw.close()
                
                fhw=open(options.output_directory+"/"+correlation_type+"_threshold_"+str(threshold)+"_inflation_"+str(inflation)+".davies_bouldin_score","w")
                fhw.write(str(davies_bouldin_score)+"\n")
                fhw.close()

def performMarkovClusteringMain(options,logger_proxy,logging_mutex):
    performMarkovClustering(options,logger_proxy,logging_mutex)
    #computeModularityForEachClustering(options,logger_proxy,logging_mutex)
    #selectBestClusters(options,logger_proxy,logging_mutex)

def main():
    commandLineArg=sys.argv
    if len(commandLineArg)==1:
        print("Please use the --help option to get usage information")
    
    options=parseCommandLineArguments()
    logger_proxy,logging_mutex=configureLogger(options)
    
    validateCommandLineArguments(options,logger_proxy,logging_mutex)
    with logging_mutex:
        logger_proxy.info("Validation of Command Line argument completed: validateCommandLineArguments(options,logger) execution successful")
    
    readInputData(options,logger_proxy,logging_mutex)
    with logging_mutex:
        logger_proxy.info("Reading of input data and creation of pandas dataframe completed: readInputData(options,logger) execution successful")
    
    performMarkovClusteringMain(options,logger_proxy,logging_mutex)
    
if __name__ == "__main__":
    main()