library("NbClust")
raw_data <- read.table("raw_tpm_counts_june_08_2020.tsv", header = TRUE, sep = "", quote = "\"")
data_to_be_clustered <-raw_data[,2:5211]
nb <- NbClust(data_to_be_clustered, method = "kmeans", diss = NULL, distance = "euclidean", index = "all", alphaBeale = 0.1)
nb$All.index
nb$Best.nc
nb$All.CriticalValues
nb$Best.partition
