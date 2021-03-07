################################################################################################################################################################
# 
# 
# 
# 
# 
# 
# 
# 
################################################################################################################################################################


"""
Command to run 
Pronto

nohup python /work/LAS/mash-lab/bhandary/regulon_prediction/predict_regulons.py \
-pb /work/LAS/mash-lab/bhandary/analysis_regulon_prediction/probe_sets_with_regulon_for_microarray.csv \
-o /work/LAS/mash-lab/bhandary/analysis_regulon_prediction/predict_regulons \
-n 170 \
-gtf /work/LAS/mash-lab/bhandary/data/arath/transcriptome/Arabidopsis_thaliana.TAIR10.43.modified.gtf \
-gm /work/LAS/mash-lab/bhandary/analysis_regulon_prediction/genes_in_microarray \
-g /work/LAS/mash-lab/bhandary/data/arath/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa \
-c /work/LAS/mash-lab/bhandary/analysis_regulon_prediction/raw_tpm_counts_june_08_2020.tsv \
-star_index /work/LAS/mash-lab/bhandary/analysis_regulon_prediction/predict_regulons/star_index \
-map /work/LAS/mash-lab/bhandary/data/arath/transcriptome/only_protein_coding_transcriptome_to_gene_map \
-m /work/LAS/mash-lab/bhandary/analysis_regulon_prediction/genes_to_microarray \
1> /work/LAS/mash-lab/bhandary/analysis_regulon_prediction/predict_regulons.output \
2> /work/LAS/mash-lab/bhandary/analysis_regulon_prediction/predict_regulons.error &

Ceres/Atlas

nohup python /project/maizegdb/sagnik/bhandary/regulon_prediction/predict_regulons.py \
-pb /project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/probe_sets_with_regulon_for_microarray.csv \
-o /project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/predict_regulons \
-n 250 \
-gtf /project/maizegdb/sagnik/data/arath/transcriptome/Arabidopsis_thaliana.TAIR10.43.modified.gtf \
-g /project/maizegdb/sagnik/data/arath/genome//Arabidopsis_thaliana.TAIR10.dna.toplevel.fa \
-gm /project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/genes_in_microarray \
-c /project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/raw_tpm_counts_june_08_2020.tsv \
-star_index /project/maizegdb/sagnik/data/arath/transcriptome/star_index_transcriptome \
-map /project/maizegdb/sagnik/bhandary/data/arath/transcriptome/only_protein_coding_transcriptome_to_gene_map \
1> /project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/predict_regulons.output \
2> /project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/predict_regulons.error &
"""

import argparse
import csv
import glob
import multiprocessing
import os
import pickle
import pprint
import random
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score,
                             recall_score)
from sklearn.metrics.ranking import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report


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

def readFromProbeSetFile(options):
    """
    Prepare gene to regulon file
    """
    if os.path.exists(options.output+"/genes_to_regulons.tsv")==True:return
    # Find protein coding genes from gtf file
    list_of_protein_coding_genes=[]
    fhr=open(options.gtf,"r")
    for line in fhr:
        if line.strip().split("\t")[2]=="CDS":
            for ele in line.strip().split("\t")[-1].split(";"):
                if "gene_id" in ele:
                    list_of_protein_coding_genes.append(ele.strip().split()[-1].strip("\""))
    fhr.close()
    list_of_protein_coding_genes=list(set(list_of_protein_coding_genes))
    
    gene_to_regulon = {}
    fhr=open(options.probe_set_info,"r")
    csv_reader = csv.reader(fhr,delimiter = ",")
    for row in csv_reader:
        if "Locus ID" in row:continue
        probe,gene,regulon_num,regulon_name = row[:4]
        gene=gene.upper()
        regulon_name=regulon_name.strip("\"")
        if regulon_name=="" or regulon_name==" ":
            regulon_name="X"
        #print(probe,gene,regulon_num,regulon_name)
        #print("Gene = ",gene,"Regulon name = ",regulon_name)
        if "," in gene:
            all_genes = gene.split(",")
            for gene in all_genes:
                if gene not in gene_to_regulon:
                    gene_to_regulon[gene]=[]
                gene_to_regulon[gene].append(regulon_name)
        else:
            if gene in set(list_of_protein_coding_genes):
                if gene not in gene_to_regulon:
                    gene_to_regulon[gene]=[]
                gene_to_regulon[gene].append(regulon_name)
    fhr.close()
    
    fhr=open(options.genes_in_microarray,"r")
    for line in fhr:
        genes = line.strip().split(";")
        for gene in genes:
            gene=gene.split(".")[0]
            if gene not in gene_to_regulon:
                gene_to_regulon[gene] = []
                gene_to_regulon[gene].append("X")
    fhr.close()
    
    fhw = open(options.output+"/genes_to_regulons.tsv","w")
    for gene in gene_to_regulon:
        if len(gene_to_regulon[gene]) == 1:
            fhw.write(gene+"\t"+gene_to_regulon[gene][0]+"\n")
    fhw.close()

def findTechnicallyCorrelatedGenePairs(options):
    
    list_of_protein_coding_genes=[]
    fhr=open(options.gtf,"r")
    for line in fhr:
        if line.strip().split("\t")[2]=="CDS":
            for ele in line.strip().split("\t")[-1].split(";"):
                if "gene_id" in ele:
                    list_of_protein_coding_genes.append(ele.strip().split()[-1].strip("\""))
    fhr.close()
    list_of_protein_coding_genes=list(set(list_of_protein_coding_genes))
    
    gene_to_regulon = {}
    for line in open(options.output+"/genes_to_regulons.tsv").read().split("\n")[::-1]:
        if len(line)<5:continue
        gene,regulon = line.split("\t")
        gene_to_regulon[gene]=regulon
    
    # Select only those transcripts that are represented in the probe sets
    cmd="gffread "+options.gtf
    cmd+=" -g "+options.genome
    cmd+=" -w "+options.output+"/all_transcripts.fasta "
    os.system(cmd)
    
    cmd="perl -pe '/^>/ ? print \"\\n\" : chomp' "
    cmd+=options.output+"/all_transcripts.fasta "
    cmd+=" | tail -n +2 > "
    cmd+=options.output+"/temp "
    os.system(cmd)
    
    cmd="mv "+options.output+"/temp "+options.output+"/all_transcripts.fasta "
    os.system(cmd)
    
    fhw=open(options.output+"/sub_transcripts.fasta","w")
    fhr=open(options.output+"/all_transcripts.fasta","r")
    for line in fhr:
        if line[0]==">":
            transcript = line.split()[0][1:]
            gene = transcript.split(".")[0]
            #print(gene)
            if gene not in gene_to_regulon or gene not in list_of_protein_coding_genes:continue
            fhw.write(">"+transcript+"\n"+fhr.readline().strip()+"\n")
    fhr.close()
    fhw.close()
    
    os.system("mkdir -p "+options.output+"/indices")
    cmd="makeblastdb "
    cmd+=" -in "+options.output+"/sub_transcripts.fasta"
    cmd+=" -dbtype nucl "
    cmd+=" -out "+options.output+"/indices/blast_db"
    os.system(cmd)
    
    cmd="blastn "
    cmd+=" -query "+options.output+"/sub_transcripts.fasta"
    cmd+=" -db "+options.output+"/indices/blast_db"
    cmd+=" -out "+options.output+"/sub_transcripts.blast "
    cmd+=" -outfmt \"6 qseqid sseqid pident qcov evalue bitscore\" "
    cmd+=" -word_size 151 "
    cmd+=" -num_threads "+str(options.cpu)
    os.system(cmd)
    
    highly_similar_gene_pairs = []
    fhr=open(options.output+"/sub_transcripts.blast","r")
    for line in fhr:
        #print("line",line)
        query,subject,pident,qcov,evalue=line.split()
        #print(query,subject,pident,qcov,evalue)
        query=query.split(".")[0]
        subject=subject.split(".")[0]
        if query!=subject:
            highly_similar_gene_pairs.append("\t".join(sorted([query,subject])))
    fhr.close()
    highly_similar_gene_pairs=list(set(highly_similar_gene_pairs))

def findSamplesToDownload(options):
    
    options.alignments=options.output+"/alignments"
    highly_similar_gene_pairs = []
    fhr=open(options.output+"/sub_transcripts.blast","r")
    for line in fhr:
        #print("line",line)
        query,subject,pident,qcov,evalue=line.split()
        #print(query,subject,pident,qcov,evalue)
        query=query.split(".")[0]
        subject=subject.split(".")[0]
        if query!=subject:
            highly_similar_gene_pairs.append("\t".join(sorted([query,subject])))
    fhr.close()
    highly_similar_gene_pairs=list(set(highly_similar_gene_pairs))
    
    all_genes = []
    for row in highly_similar_gene_pairs:
        all_genes.append(row.split("\t")[0])
        all_genes.append(row.split("\t")[1])
    
    gene_to_regulon = {}
    for line in open(options.output+"/genes_to_regulons.tsv").read().split("\n")[::-1]:
        if len(line)<5:continue
        gene,regulon = line.split("\t")
        gene_to_regulon[gene]=regulon
    
    """for gene in all_genes:
        print(gene,gene_to_regulon[gene])"""
        
    #print(all_genes)
    count_data = pd.io.parsers.read_csv(options.counts,delimiter="\t",index_col=0)
    #print(count_data)
    #print(count_data.T.idxmax())
    samples_to_be_tested = []
    max_df = count_data.T.idxmax()
    for gene in all_genes:
        #if os.path.exists(options.alignments+"/"+max_df.loc[gene]+"_STAR_Aligned.out.bam")==False:
        samples_to_be_tested.append(max_df.loc[gene])
    fhw=open(options.output+"/download_these","w")
    fhw.write("\n".join(samples_to_be_tested))
    fhw.close()

def calculateJaccardIndexInParallel(eachinput):
    highly_similar_gene_pairs,options,sample_name,genes_of_interest=eachinput
    cmd="samtools view -@ 1 "+options.alignments+"/"+sample_name+"_STAR_Aligned.out.bam > "
    cmd+=options.alignments+"/"+sample_name+"_STAR_Aligned.out.sam"
    os.system(cmd)
        
    gene_to_reads = {}
    fhr=open(options.alignments+"/"+sample_name+"_STAR_Aligned.out.sam","r")
    for line in fhr:
        if line[0]=="#":continue
        gene = line.strip().split()[2].split(".")[0]
        read = line.strip().split()[0]
        """print(gene,read)
        sys.stdout.flush()"""
        if gene not in genes_of_interest:continue 
        if gene not in gene_to_reads:
            gene_to_reads[gene]=[]
        gene_to_reads[gene].append(read)
    fhr.close()
    
    #print(gene_to_reads)
    for gene in gene_to_reads:
        gene_to_reads[gene]=list(set(gene_to_reads[gene]))
    
    
    jaccard_index={}
    for row in highly_similar_gene_pairs:
        gene1,gene2=row.split("\t")
        if gene1 not in gene_to_reads or gene2 not in gene_to_reads:continue
        if gene1+"-"+gene2 not in jaccard_index:
            if len(set(gene_to_reads[gene1])&set(gene_to_reads[gene2]))<10:continue
            jaccard_index[gene1+"-"+gene2] = {}
            jaccard_index[gene1+"-"+gene2] = len(set(gene_to_reads[gene1])&set(gene_to_reads[gene2]))/len(set(gene_to_reads[gene1])|set(gene_to_reads[gene2]))
    os.system("rm "+options.alignments+"/"+sample_name+"_STAR_Aligned.out.sam")
    print(sample_name,"processed")
    sys.stdout.flush()
    return jaccard_index,sample_name

def readCountsFile(counts_filename,genes_to_be_selected):
    countsdata = {}
    fhr=open(counts_filename,"r")
    for line in fhr:
        if "Gene" in line:continue
        gene, gene_counts = line.strip().split()[0], list(map(float,line.strip().split()[1:]))
        if gene not in genes_to_be_selected:continue
        countsdata[gene]=gene_counts
    fhr.close()
    return countsdata

def performPCA(countsdata):
    countsdata_matrix = np.array([np.array(countsdata[gene]) for gene in countsdata])
    countsdata_matrix_pca_all_components = PCA(n_components=len(countsdata_matrix[0])).fit_transform(countsdata_matrix)
    return countsdata_matrix_pca_all_components


def performTrainingAndTesting(X_train_normalized_pca, X_test_normalized_pca, Y_train, Y_test, options, pca_n_comp, machine_learner,regulon,dataset):
    #return machine_learner,1,2,3,4,5,6,7
    #pickle_filename=options.pickle_file_directory+"/"+options.regulon_name+"_"+machine_learner+"_pca_"+pca_n_comp+".pickle"
    #print ("Y_train_0=", Y_train.count(0), "Y_train_1=", Y_train.count(1), "Y_test_0=", Y_test.count(0), "Y_test_1=", Y_test.count(1))
    n_classes = 2
    options.pickle_file_directory = options.output+"/pickle_files"
    os.system("mkdir -p "+options.pickle_file_directory)
    pickle_filename=options.pickle_file_directory+"/"+"_".join(["PCA"+str(pca_n_comp),"regulon",
                                                                regulon,"dataset",dataset,
                                                                machine_learner])+".pickle"
                                                                
    currently_processing_filename=options.pickle_file_directory+"/"+"_".join(["PCA"+str(pca_n_comp),"regulon",
                                                                regulon,"dataset",dataset,
                                                                machine_learner])+".currently_processing"
                                                                
    if os.path.exists(pickle_filename)==False and os.path.exists(currently_processing_filename)==False:
        os.system("touch \""+currently_processing_filename+"\"")
        print("Performing cross validation for "+machine_learner+" Number of PCA components: "+pca_n_comp+" Regulon: "+regulon+" Dataset: "+dataset)
        sys.stdout.flush()
        if machine_learner=="svm":
            clf = svm.SVC(probability=True)
            tuned_parameters=[{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 50, 100, 500, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 50, 100, 500, 1000]}]
        elif machine_learner=="sgd":
            clf = SGDClassifier(random_state=42)
            tuned_parameters = {
            'loss' : ['hinge','log','modified_huber','squared_hinge'],
            'penalty' : ['l2','l1','elasticnet'],
            'alpha' : [1e-2, 1e-4, 1e-8]}
        elif machine_learner=="rf":
            clf = RandomForestClassifier(random_state=42)
            tuned_parameters = {
            'n_estimators' : [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            'max_features' : ['auto', 'sqrt', 'log2'],
            'max_depth' : [5,10, 15, 20, 25, 30, 35, 40, 45, 50],
            'criterion' : ['gini', 'entropy']}
        elif machine_learner=="mlp":
            clf = MLPClassifier(random_state=42)
            tuned_parameters = {
            'solver' : ['lbfgs','sgd','adam'],
            'alpha' : [1e-3, 1e-4, 1e-5, 1e-6],
            'hidden_layer_sizes' : [(10,),(20,),(30,),(40,),(50,),(60,),(70,),(80,),(90,),(250,),(500,)],
            'activation' : ['identity','logistic','tanh','relu']}
        elif machine_learner=="gbt":
            clf = GradientBoostingClassifier(random_state=42)
            tuned_parameters = {
            'n_estimators' : [75, 100, 125, 150, 175, 200, 225, 275, 300],
            'learning_rate' : [1, 0.5, 0.25, 0.1, 0.05, 0.01],
            'max_features' : ['auto', 'sqrt', 'log2']
            }
        elif machine_learner=="lda":
            clf = LinearDiscriminantAnalysis()
            tuned_parameters = {
            'solver' : ['svd','lsqr','eigen'],
            'tol' : [1e-4, 1e-3, 1e-2]}
        CV_clf = GridSearchCV(clf, tuned_parameters, cv=5, n_jobs=-1,scoring='f1_macro',verbose=100)
        CV_clf.fit(X_train_normalized_pca,Y_train)
        #pprint.pprint(CV_clf.cv_results_)
        cross_validation_output=options.pickle_file_directory+"/"+"_".join(["PCA"+str(pca_n_comp),"regulon",
                                                                regulon,"dataset",dataset,
                                                                machine_learner])+".cross_validation_output"
        cross_validation_results=pd.concat([pd.DataFrame(CV_clf.cv_results_["params"]),pd.DataFrame(CV_clf.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
        cross_validation_results.to_csv(cross_validation_output)
        print("="*200)
        sys.stdout.flush()
        pickle.dump(CV_clf,open(pickle_filename,"wb"))
        os.system("rm \""+currently_processing_filename+"\"")
    else:
        CV_clf=pickle.load(open(pickle_filename,"rb"))
        Y_pred = CV_clf.predict(X_test_normalized_pca)
        CV_clf_proba = CV_clf.predict_proba(X_test_normalized_pca)
        precision = {}
        recall = {}
        thresholds = {}
        Y_test_proba = []
        #print (CV_clf_proba.shape)
        #print (Y_test)
        #print (Y_test[:,1])
        #print (Y_test_proba)
        #print (len(Y_test))
        i = 0
        #print (Y_test)
        while i <len(Y_test):
            if Y_test[i] == '0':
                Y_test_proba.append([1,0,0])
            elif Y_test[i] == '1':
                Y_test_proba.append([0,1,0])
            elif Y_test[i] == '2':
                Y_test_proba.append([0,0,1])
            i+=1
        #print (Y_test_proba)
        Y_test_proba_array = np.array(Y_test_proba)
        #print (Y_test_proba_array)
        for j in range(n_classes):
            print (f"Class {j}")
            precision[j], recall[j], thresholds[j] = precision_recall_curve(Y_test_proba_array[:,j], CV_clf_proba[:,j])
            precision[j]=list(precision[j])
            recall[j]=list(recall[j])
            thresholds[j]=list(thresholds[j])
        #print ("Inside performTrainingAndTesting",precision, recall, thresholds," Size: ",len(Y_test_proba_array),len(CV_clf_proba))
        #report = classification_report(Y_test_proba_array, CV_clf_proba, output_dict=True)
        """precision, recall, _ = precision_recall_curve(Y_test, CV_clf_proba[:,2])"""
        """auc_precision_recall = auc(recall, precision)
        precision = precision_score(Y_test, Y_pred, average = 'binary')
        recall = recall_score(Y_test, Y_pred, average = 'binary')
        f1_test_score = f1_score(Y_test, Y_pred, average = 'binary')
        matthews_corr = matthews_corrcoef(Y_test, Y_pred)"""
        #ROC_score = roc_auc_score(Y_test, Y_pred) 
        
        
        #return machine_learner,precision,recall,f1_test_score,matthews_corr,ROC_score,list(Y_pred).count(0),list(Y_pred).count(1)
        #return machine_learner,precision,recall,f1_test_score,matthews_corr,list(Y_pred).count(0),list(Y_pred).count(1)
        #return machine_learner,precision,recall,f1_test_score,matthews_corr,auc_precision_recall,list(Y_pred).count(0),list(Y_pred).count(1), CV_clf_proba
        #return machine_learner,precision,recall,f1_test_score,matthews_corr,list(Y_pred).count(0),list(Y_pred).count(1), CV_clf_proba
        
        return machine_learner,precision,recall, thresholds, regulon
        """return machine_learner,-1,-1,-1,-1,-1,-1,-1,-1
        if os.path.exists(pickle_filename)==False:
            CV_clf = None
        else:
            print("Loading trained model for "+machine_learner+". Number of PCA components: "+pca_n_comp+". Regulon: "+regulon+" Dataset: "+dataset)
            CV_clf=pickle.load(open(pickle_filename,"rb"))
    
    if CV_clf is not None:
        Y_pred = CV_clf.predict(X_test_normalized_pca)
        CV_clf_proba = CV_clf.predict_proba(X_test_normalized_pca)
        precision, recall, _ = precision_recall_curve(Y_test, CV_clf_proba[:,1])
        auc_precision_recall = auc(recall, precision)
        precision = precision_score(Y_test, Y_pred, average = 'binary')
        recall = recall_score(Y_test, Y_pred, average = 'binary')
        f1_test_score = f1_score(Y_test, Y_pred, average = 'binary')
        matthews_corr = matthews_corrcoef(Y_test, Y_pred)
        #ROC_score = roc_auc_score(Y_test, Y_pred)
        
        print (CV_clf_proba)
        #return machine_learner,precision,recall,f1_test_score,matthews_corr,ROC_score,list(Y_pred).count(0),list(Y_pred).count(1)
        #return machine_learner,precision,recall,f1_test_score,matthews_corr,list(Y_pred).count(0),list(Y_pred).count(1)
        return machine_learner,precision,recall,f1_test_score,matthews_corr,auc_precision_recall,list(Y_pred).count(0),list(Y_pred).count(1), CV_clf_proba
    else:
        return machine_learner,-1,-1,-1,-1,-1,-1,-1,-1"""

def alignReads(options):
    # Align reads
    options.alignments=options.output+"/alignments"
    os.system("mkdir -p "+options.alignments)
    options.salmon_counts=options.output+"/salmon_counts"
    os.system("mkdir -p "+options.salmon_counts)
    options.raw_data=options.output+"/raw_data"
    for sample_name in open(options.output+"/download_these").read().split("\n"):
        cmd="STAR "
        cmd+=" --genomeDir "+options.star_index
        cmd+=" --runThreadN "+str(options.cpu)
        cmd+=" --readFilesIn "+options.raw_data+"/"+sample_name+"_1.fastq "+options.raw_data+"/"+sample_name+"_2.fastq "
        cmd+=" --outSAMtype BAM Unsorted "
        cmd+=" --outFilterMultimapNmax 500 " 
        cmd+=" --outFilterMismatchNmax 5  " 
        cmd+=" --alignIntronMin 1  "
        cmd+=" --alignIntronMax 1 "
        cmd+=" --limitBAMsortRAM 107374182400"
        cmd+=" --genomeLoad LoadAndKeep "
        cmd+=" --outFilterMatchNminOverLread 0.95 "
        cmd+=" --outFileNamePrefix "+options.alignments+"/"+sample_name+"_STAR_"
        cmd+=" > "+options.alignments+"/"+sample_name+"_STAR.output"
        cmd+=" 2> "+options.alignments+"/"+sample_name+"_STAR.error"
        if os.path.exists(options.alignments+"/"+sample_name+"_STAR_Aligned.out.bam")==False:
            os.system(cmd)

def computeJaccardIndex(options):
    # Calculate Jaccard Index
    
    if os.path.exists(options.output+"/jaccard_index.pkl")==False:
        highly_similar_gene_pairs = []
        genes_of_interest = []
        fhr=open(options.output+"/sub_transcripts.blast","r")
        for line in fhr:
            #print("line",line)
            query,subject,pident,qcov,evalue=line.split()
            #print(query,subject,pident,qcov,evalue)
            query=query.split(".")[0]
            subject=subject.split(".")[0]
            if query!=subject:
                highly_similar_gene_pairs.append("\t".join(sorted([query,subject])))
                genes_of_interest.append(subject)
                genes_of_interest.append(query)
        fhr.close()
        highly_similar_gene_pairs=list(set(highly_similar_gene_pairs))
        genes_of_interest=set(genes_of_interest)
        
        
        pool = multiprocessing.Pool(processes=int(options.cpu))
        jaccard_index = {}
        allinputs = []
        all_samples = list(set(open(options.output+"/download_these").read().split("\n")))
        for sample_name in all_samples:
            allinputs.append([highly_similar_gene_pairs,options,sample_name,genes_of_interest])
        
        print(len(allinputs))
        sys.stdout.flush()
        for row in pool.map(calculateJaccardIndexInParallel,allinputs):
            jaccard_index_per_sample,sample_name=row
            for gene_pair in jaccard_index_per_sample:
                if gene_pair not in jaccard_index:
                    jaccard_index[gene_pair]={}
                jaccard_index[gene_pair][sample_name]=jaccard_index_per_sample[gene_pair]
            
        pickle.dump(jaccard_index,open(options.output+"/jaccard_index.pkl","wb"))
    
    #print("Inside here")
    sys.stdout.flush()
    jaccard_indices=[]
    jaccard_index = pickle.load(open(options.output+"/jaccard_index.pkl","rb"))
    jaccard_index_for_each_gene_pair={}
    for each_gene_pair in jaccard_index:
        #print(each_gene_pair,jaccard_index[each_gene_pair])
        jaccard_index_for_each_gene_pair[each_gene_pair] = max([jaccard_index[each_gene_pair][key] for key in jaccard_index[each_gene_pair]])
        #print(each_gene_pair,jaccard_index_for_each_gene_pair[each_gene_pair])
        jaccard_indices.append(jaccard_index_for_each_gene_pair[each_gene_pair])
    #print(np.percentile(sorted(jaccard_indices),[25,50,75,90]))
    threshold_for_similar_genes = 0.8
    set_of_similar_genes = []
    dict_of_similar_genes = {}
    for each_pair in jaccard_index_for_each_gene_pair:
        if jaccard_index_for_each_gene_pair[each_pair]>=threshold_for_similar_genes:
            set_of_similar_genes.append([each_pair.split("-")[0],each_pair.split("-")[1]])
            dict_of_similar_genes[each_pair.split("-")[0]] = each_pair.split("-")[1]
            dict_of_similar_genes[each_pair.split("-")[1]] = each_pair.split("-")[0]
    #pprint.pprint(set_of_similar_genes)
    return set_of_similar_genes,dict_of_similar_genes
    
def main():
    commandLineArg=sys.argv
    if len(commandLineArg)==1:
        print("Please use the --help option to get usage information")
    options=parseCommandLineArguments()
    
    os.system("mkdir -p "+options.output)
    readFromProbeSetFile(options)
    """
    findTechnicallyCorrelatedGenePairs(options)
    
    findSamplesToDownload(options)
    
    # Download samples 
    cmd="python /work/LAS/mash-lab/bhandary/regulon_prediction/download_and_dump_fastq_from_SRA.py "
    cmd+=" -s "+options.output+"/download_these "
    cmd+=" -o "+options.output+"/raw_data "
    cmd+=" -n "+options.cpu
    os.system(cmd)
    
    alignReads(options)
    """
    set_of_similar_genes,dict_of_similar_genes = computeJaccardIndex(options)
    
    #pprint.pprint(set_of_similar_genes)
    # Split data into test and train
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
    
    labels_to_genes = {}
    for gene in gene_to_regulon:
        if gene in set( [item for sublist in set_of_similar_genes for item in sublist]):
            similar = 1
            # Check if the related gene is in the same regulon
            """print(gene,dict_of_similar_genes[gene],gene_to_regulon[gene],gene_to_regulon[dict_of_similar_genes[gene]])
            sys.stdout.flush()"""
        else:
            similar = 0
        label = gene_to_regulon[gene]
        if label not in labels_to_genes:
            labels_to_genes[label]=[[],[]]
        labels_to_genes[label][similar].append(gene)
    """
    for label in labels_to_genes:
        print(label,len(labels_to_genes[label][0]),len(labels_to_genes[label][1]))
    """
    options.testtrain = options.output+"/testtrain"
    #os.system("rm -rf "+options.testtrain)
    os.system("mkdir -p "+options.testtrain)
    testtrain_split=0.75
    num_of_datsets = 20
    
    for i in range(num_of_datsets):
        for label in labels_to_genes:
            modified_regulon_name = labels_to_regulons[label].replace(",","_").replace(" ","").replace("/","_").strip()
            if len(labels_to_genes[label][0])<20:continue
            if os.path.exists(f"{options.testtrain}/testtrain_{i+1}_reg_{modified_regulon_name}")==True:continue
            fhw = open(f"{options.testtrain}/testtrain_{i+1}_reg_{modified_regulon_name}","w")
            all_genes_in_positive_dataset,all_genes_in_negative_dataset,all_genes_in_non_positive_dataset = [], [], []
            all_genes_in_positive_dataset.extend(labels_to_genes[label][0])
            for l in labels_to_genes:
                if l==label:continue
                if labels_to_regulons[l].strip()=="X":
                    all_genes_in_negative_dataset.extend(labels_to_genes[l][0])
                else:
                    all_genes_in_non_positive_dataset.extend(labels_to_genes[l][0])
                
            random.shuffle(all_genes_in_positive_dataset)
            training_genes_pos = all_genes_in_positive_dataset[:int(len(all_genes_in_positive_dataset)*testtrain_split)]
            testing_genes_pos = all_genes_in_positive_dataset[int(len(all_genes_in_positive_dataset)*testtrain_split):]
            
            random.shuffle(all_genes_in_negative_dataset)
            training_genes_neg = all_genes_in_negative_dataset[:int(len(all_genes_in_negative_dataset)*testtrain_split)]
            testing_genes_neg = all_genes_in_negative_dataset[int(len(all_genes_in_negative_dataset)*testtrain_split):]
            
            random.shuffle(all_genes_in_non_positive_dataset)
            training_genes_non_pos = all_genes_in_non_positive_dataset[:int(len(all_genes_in_non_positive_dataset)*testtrain_split)]
            testing_genes_non_pos = all_genes_in_non_positive_dataset[int(len(all_genes_in_non_positive_dataset)*testtrain_split):]
            
            fhw.write("Train\t1\t"+",".join(training_genes_pos)+"\n")
            fhw.write("Train\t2\t"+",".join(training_genes_non_pos)+"\n")
            fhw.write("Train\t0\t"+",".join(training_genes_neg)+"\n")
            
            fhw.write("Test\t1\t"+",".join(testing_genes_pos)+"\n")
            fhw.write("Test\t2\t"+",".join(testing_genes_non_pos)+"\n")
            fhw.write("Test\t0\t"+",".join(testing_genes_neg)+"\n")
            
            #print(modified_regulon_name,len(all_genes_in_positive_dataset),len(all_genes_in_non_positive_dataset),len(all_genes_in_negative_dataset))
            fhw.close()
    #return
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
    #print(len(countsdata_matrix_pca_all_components))
    # Prepare datasets for train and test
    options.datasets = options.output+"/datasets"
    #os.system("rm -rf "+options.datasets)
    os.system("mkdir -p "+options.datasets)
    num_of_datsets = 5
    output_file = open(options.output+".tsv", 'w')
    for i in range(num_of_datsets):
        for label in labels_to_genes:
            modified_regulon_name = labels_to_regulons[label].replace(",","_").replace(" ","").replace("/","_").strip()
            """skip=0
            for file in glob.glob(options.output+"/pickle_files/*"):
                #print(modified_regulon_name,file,modified_regulon_name in file)
                if modified_regulon_name in file:
                    skip=1
                    break
            if skip==1:continue"""
            #print("Dothis",modified_regulon_name)
            #continue
            if modified_regulon_name=="X":continue
            if len(labels_to_genes[label][0])<20:continue
            fhr = open(f"{options.testtrain}/testtrain_{i+1}_reg_{modified_regulon_name}","r")
            #fhw_train=open(f"{options.datasets}+/training_{i+1}_reg_{modified_regulon_name}","w")
            #fhw_test=open(f"{options.datasets}+/testing_{i+1}_reg_{modified_regulon_name}","w")
            genes_in_each_class = {"Train":{},"Test":{}}
            for line in fhr:
                model_set,label,genes = line.strip().split("\t")
                #print(genes_in_each_class[model_set])
                sys.stdout.flush()
                genes_in_each_class[model_set][label] = [gene.strip() for gene in genes.split(",")]
    
            #fhw_train.close()
            #fhw_test.close()        
            complete_info={}
            for pca_comp in list(range(0,501,25))[1:]:
                X_train = []
                X_test = []
                Y_train = []
                Y_test = []
                for label in genes_in_each_class["Train"]:
                    for gene in genes_in_each_class["Train"][label]:
                        if gene not in countsdata_matrix_pca_all_components:continue
                        X_train.append(np.array(countsdata_matrix_pca_all_components[gene][:pca_comp]))
                        Y_train.append(label)
                for label in genes_in_each_class["Test"]:
                    for gene in genes_in_each_class["Test"][label]:
                        if gene not in countsdata_matrix_pca_all_components:continue
                        X_test.append(np.array(countsdata_matrix_pca_all_components[gene][:pca_comp]))
                        Y_test.append(label)
                #pprint.pprint(X_train)
                X_train=np.array(X_train)
                X_test=np.array(X_test)
                scaler=preprocessing.StandardScaler().fit(X_train)
                X_train_normalized = scaler.transform(X_train)
                X_test_normalized = scaler.transform(X_test)
                #for machine_learner in ["sgd","mlp","rf","gbt","svm"]:
                for machine_learner in ["mlp","rf","gbt"]:
                    
                    machine_learner,precision,recall,thresholds,regulon = performTrainingAndTesting(X_train_normalized, X_test_normalized, Y_train, Y_test, options, str(pca_comp),machine_learner,modified_regulon_name,str(i+1))
                    #df = pd.dataframe(report).transpose
                    #df.to_csv(options.output+"pr_curves_for_r_"+regulon+"_"+str(pca_comp)+".csv")
                    #complete_info[machine_learner]=[round(x,2) for x in [precision,recall,f1_test_score,matthews_corr,ROC_score,Y_pred_1]]
                    print(len(precision),len(recall),len(thresholds),regulon,pca_comp,machine_learner)
                    sys.stdout.flush()
                    for class_i in precision:
                        print(class_i,regulon,pca_comp,len(precision[class_i]),len(recall[class_i]),len(thresholds[class_i]))
                        for j in range(len(precision[class_i])-1):
                            line_to_be_written_to_file=f"{precision[class_i][j]}\t{recall[class_i][j]}\t{thresholds[class_i][j]}\t{class_i}\t{regulon}\t{pca_comp}\t{machine_learner}\t{i+1}"
                            #output_file.write(str(precision[i])+"\t"+str(recall[i])+"\t"+str(thresholds[i])+"\t"+regulon+"\t"+str(pca_comp)+"\t"+machine_learner+"\t"+str(i)+"\n")
                            output_file.write(line_to_be_written_to_file)
    
                    #print (precision, recall,thresholds,regulon)
                    #os.sys("Rscript /project/maizegdb/sagnik/bhandary/regulon_prediction/figure_with_regulons_PR_all_pcas.R "+precision+" "+recall+" "+thresholds)
                    #complete_info[machine_learner]=[round(x,2) for x in [precision,recall,f1_test_score,matthews_corr,Y_pred_1]]
            #complete_info_pd=pd.DataFrame.from_dict(complete_info,orient='index')
            #complete_info_pd.columns = ["precision","recall","f1","mcc","pos"]
            print(modified_regulon_name)
            #pprint.pprint(complete_info_pd)
            sys.stdout.flush()
    
if __name__ == "__main__":
    main()
