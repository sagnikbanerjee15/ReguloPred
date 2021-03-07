import os, sys

for k in range(125, 501, 25):
    new_file = "/project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/pbs_scripts/run_mcl_pca_"+str(k)+".pbs"
    fhw = open(new_file, "w")
    fhw.write("""#!/bin/bash

#SBATCH -N 10
#SBATCH -n 250
#SBATCH --time=14-00:00:00
#SBATCH --mail-user=bhandary@iastate.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --error=/project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/mcl_pca_"""+str(k)+""".error
#SBATCH --output=/project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/mcl_pca_"""+str(k)+""".output
#SBATCH -A maizegdb

conda activate regulon_prediction


python /project/maizegdb/sagnik/bhandary/regulon_prediction/generate_MCL_clusters.py \
--gene_counts /project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/raw_tpm_counts_june_08_2020.tsv \
--output_directory /project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/predict_regulons \
--cpu 250 \
--theshold_adjacency_matrix 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 \
--pca """+str(k)+""" \
1> /project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/mcl_pca_"""+str(k)+""".output \
2> /project/maizegdb/sagnik/bhandary/analysis_regulon_prediction/mcl_pca_"""+str(k)+""".error""")