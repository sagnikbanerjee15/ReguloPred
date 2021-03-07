import argparse
import os
import pprint
import sys

import pandas as pd
from ruffus.proxy_logger import *


def parseCommandLineArguments():
    parser = argparse.ArgumentParser(prog="download_align_quantify_genes.py",description="Downloads, aligns and quantifies the genes")
    
    # Mandatory arguments
    parser.add_argument("--transcriptome_fasta","-t",help="Enter the transcripts file in fasta format", required = True)
    parser.add_argument("--output_directory","-o",help="Enter the location of the output directory", required = True)
    parser.add_argument("--list_of_sample_names","-l",help="Enter the list of sample names to be processed", required = True)
    parser.add_argument("--star_index","-star_index",help="Enter the location of STAR index",required = True)
    parser.add_argument("--transcript_to_gene_map","-map",help="Enter the transcript to gene map",required = True)
    
    # Optional arguments
    parser.add_argument("--cpu","-n",help="Enter the number of CPUs",default = 1)
    parser.add_argument("--process_first_n_samples","-p",help="Indicate the total number of samples to be processed. Argument is included for exploratory purposes. Default action is to process everything",default = -1)
    parser.add_argument("--skip_clean","-sc",help="Set this parameter if you wish to prevent deletion of intermediate files. Please note that for a large number of samples a huge amount of intermediate data will be generated. Leave this setting to a default if you wish to avoid to hearing back from your IT department!!",default = 0)
    
    return parser.parse_args()

def configureLogger(options):
    os.system("mkdir -p "+options.output_directory)
    os.system("rm -f "+options.output_directory+"/progress.log")
    
    arguments={}
    arguments["file_name"]=options.output_directory+"/progress.log"
    arguments["formatter"] = "%(asctime)s - %(name)s - %(levelname)6s - %(message)s"
    arguments["level"]     = logging.DEBUG
    arguments["delay"]     = False
    
    (logger_proxy,logging_mutex) = make_shared_logger_and_proxy (setup_std_shared_logger,"REGPRED", arguments)
    return logger_proxy,logging_mutex

def main():
    commandLineArg=sys.argv
    if len(commandLineArg)==1:
        print("Please use the --help option to get usage information")
    options=parseCommandLineArguments()
    logger_proxy,logging_mutex=configureLogger(options)
    
    options.cpu = int(options.cpu)
    options.process_first_n_samples = int(options.process_first_n_samples)
    
    # Defining new variables
    options.raw_data = options.output_directory+"/raw_data_downloaded_from_SRA"
    options.alignments = options.output_directory+"/alignments"
    options.salmon_counts = options.output_directory+"/salmon_counts"
    options.final_counts = options.output_directory+"/final_counts"
    os.system("mkdir -p "+options.final_counts)
    
    if options.process_first_n_samples == -1:
        options.process_first_n_samples = sum([1 for line in open(options.list_of_sample_names,"r")])
        
    # Calculate batch size for efficient processing of data and optimum parallelization of the requested CPUs 
    batch_size = min(options.cpu,options.process_first_n_samples)

    all_samples = open(options.list_of_sample_names).read().split("\n")[:-1][-options.process_first_n_samples:]
        
    batch_of_samples = [all_samples[i * batch_size:(i + 1) * batch_size] for i in range((len(all_samples) + batch_size - 1) // batch_size )]
    with logging_mutex:
        logger_proxy.info("Processing samples in batches of "+str(batch_size))
        logger_proxy.info("Total samples "+str(len(all_samples)))
        logger_proxy.info("Total batches "+str(len(batch_of_samples)))
    
    samples_aligned = 0
    samples_counted = 0
    # Process each batch of samples
    for batch_no in range(len(batch_of_samples)):
        with logging_mutex:
            logger_proxy.info("Starting download of samples from batch no. "+str(batch_no+1))
            
        one_batch_of_samples = batch_of_samples[batch_no]
        fhw=open(options.output_directory+"/single_batch_of_samples","w")
        for sample_name in one_batch_of_samples:
            if os.path.exists(options.final_counts+"/"+sample_name)==False:
                fhw.write(sample_name+"\n")
        fhw.close()
        
        cmd="python /work/LAS/mash-lab/bhandary/regulon_prediction/download_and_dump_fastq_from_SRA.py "
        cmd+=" -s "+options.output_directory+"/single_batch_of_samples "
        cmd+=" -o "+options.raw_data
        cmd+=" -n "+str(options.cpu)
        os.system(cmd)
        
        with logging_mutex:
            logger_proxy.info("Finished download of samples from batch no. "+str(batch_no+1))
        
        for sample_name in open(options.output_directory+"/single_batch_of_samples").read().split("\n"):
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
            cmd+=" --outFilterScoreMinOverLread 0.95 "
            cmd+=" --outFileNamePrefix "+options.alignments+"/"+sample_name+"_STAR_"
            if os.path.exists(options.alignments+"/"+sample_name+"_STAR_Aligned.out.bam")==False and os.path.exists(options.final_counts+"/"+sample_name+".eq")==False:
                os.system(cmd)
            samples_aligned+=1
            
            with logging_mutex:
                logger_proxy.info("Alignment completed for "+sample_name+". "+str(options.process_first_n_samples-samples_aligned)+" more to go")
                
            cmd="salmon "
            cmd+=" quant "
            cmd+=" -a "+options.alignments+"/"+sample_name+"_STAR_Aligned.out.bam "
            cmd+=" -l A "
            cmd+=" -o "+options.salmon_counts+"/"+sample_name
            cmd+=" -t "+options.transcriptome_fasta
            cmd+=" -p "+str(options.cpu)
            cmd+=" --dumpEq "
            cmd+=" --dumpEqWeights " # Includes "rich" equivalence class weights in the output when equivalence class information is being dumped to file.
            cmd+=" --seqBias "
            cmd+=" --gcBias "
            #cmd+=" --posBias "
            cmd+=" -g "+options.transcript_to_gene_map
            if os.path.exists(options.salmon_counts+"/"+sample_name+"/quant.genes.sf")==False and os.path.exists(options.final_counts+"/"+sample_name+".eq")==False:
                os.system(cmd)
                cmd="cp "+options.salmon_counts+"/"+sample_name+"/quant.genes.sf "+options.salmon_counts+"/"+sample_name+".quant.genes.sf"
                os.system(cmd)
            samples_counted+=1
            
            with logging_mutex:
                logger_proxy.info("Gene count generation completed for "+sample_name+". "+str(options.process_first_n_samples-samples_counted)+" more to go")
            """    
            if os.path.exists(options.salmon_counts+"/"+sample_name+"/aux_info/eq_classes.txt")==False:
                continue
            # Process equivalence class file
            fhw = open(options.final_counts+"/"+sample_name+".eq","w")
            fhr = open(options.salmon_counts+"/"+sample_name+"/aux_info/eq_classes.txt","r")
            whole_info = fhr.read().split("\n")[:-1]
            N = int(whole_info[0])
            M = int(whole_info[1])
            # Read the transcripts # Order of transcripts is important
            transcripts = []
            whole_info_iterator = 2
            while N>0:
                transcripts.append(whole_info[whole_info_iterator])
                whole_info_iterator+=1
                N-=1
            # Read equivalence class information
            equivalence_class_info = []
            while M>0:
                equivalence_class_info.append(whole_info[whole_info_iterator].split())
                whole_info_iterator+=1
                M-=1
            fhr.close()            
            fhw.close()
            
            equivalence_class_to_transcripts = {}
            transcripts_to_equivalence_class = {}
            for eq_class_num,eachline in enumerate(equivalence_class_info):
                no_of_transcript_in_eq_class = eachline[0]
                no_of_read_in_eq_class = eachline[-1]
                eachline = eachline[1:-1]
                transcript_indices = eachline[:len(eachline)//2]
                weights = eachline[len(eachline)//2:]
                transcript_row = {}
                for j,transcript_index in enumerate(transcript_indices):
                    transcript_row[transcripts[int(transcript_index)-1]] = float(weights[j])
                    if transcripts[int(transcript_index)-1] not in transcripts_to_equivalence_class:
                        transcripts_to_equivalence_class[transcripts[int(transcript_index)-1]]={}
                    transcripts_to_equivalence_class[transcripts[int(transcript_index)-1]]["eq_class_"+str(eq_class_num)] = float(weights[j]) 
                transcript_row["Total"] = int(no_of_read_in_eq_class)
                equivalence_class_to_transcripts["eq_class_"+str(eq_class_num)] = transcript_row
            
            for key in transcripts_to_equivalence_class:
                print(key, transcripts_to_equivalence_class[key])
            for key in equivalence_class_to_transcripts:
                print(key, equivalence_class_to_transcripts[key])
            sys.exit()"""
            if options.skip_clean == 0:
                cmd="cp "+options.salmon_counts+"/"+sample_name+"/quant.genes.sf "+options.salmon_counts+"/"+sample_name+".quant.genes.sf"
                os.system(cmd)
                os.system("rm "+options.raw_data+"/"+sample_name+"*fastq")
                os.system("rm -rf "+options.alignments+"/"+sample_name+"_STAR*")
                os.system("rm -rf "+options.salmon_counts+"/"+sample_name)


if __name__ == "__main__":
    main()
