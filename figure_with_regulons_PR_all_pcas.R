library(ggplot2)
require(reshape2)
library(dplyr)
library(tidyr)
#args <- commandArgs(trailingOnly = TRUE)
setwd("~/Box Sync/analysis_regulon_prediction")
precision_recall_scores <- read.table("predict_regulons.tsv", header = FALSE, sep = "", quote = "\"")
colnames(precision_recall_scores) <- c("precision", "recall", "threshold", "class", "regulon", "pca_component", "machine_learner", "dataset_number")
prt_df <- as.data.frame(precision_recall_scores)


prt_temp_all <- prt_df %>% filter(regulon == "mixed(tricellularandmaturepollen-specific)"|regulon == "photosynthesis"| regulon == "information"| regulon == "proteinmodification_defenseresponse"| regulon == "mitosis", pca_component == "25"|  pca_component == "100"| pca_component == "200"| pca_component == "150"| pca_component == "500", class == "1")

prt_temp_all <- prt_df %>% filter(regulon == "developmentalregulation(leafapex-preferential)"|regulon == "embryomaturation(fruitandseedpreferential)"| regulon == "information(uninucleatemicrosporeandbicellularpollen-specific)"| regulon == "nuclear_otherswithverylowexpression"| regulon == "regulationoforgandevelopment", pca_component == "25"|  pca_component == "100"| pca_component == "200"| pca_component == "150"| pca_component == "500", class == "1")


prt_temp_mlp <- prt_df %>% filter(regulon == "mixed(tricellularandmaturepollen-specific)", `machine learner` == "mlp", pca_component == "25", class == "1")

for (i in prt_temp_mlp$threshold){
  p <- ggplot(prt_temp_mlp, aes(x = recall, y=precision)) + geom_line() + theme(axis.text.x = element_blank(), axis.title.x = element_blank(), axis.title.y = element_blank())
}


prt_temp_gbt <- prt_df %>% filter(regulon == "mixed(tricellularandmaturepollen-specific)",pca_component == "25", class == "1", `machine learner` == "gbt")

for (i in prt_temp_gbt$threshold){
  q <- ggplot(prt_temp_gbt, aes(x = recall, y=precision)) + geom_line() + theme(axis.text.x = element_blank(), axis.title.x = element_blank(), axis.title.y = element_blank())
}

p <- ggplot(prt_temp_all, aes(x = recall, y = precision, group = pca_component)) + geom_line(aes(colour = machine_learner)) 
p + facet_grid(pca_component ~ regulon)
