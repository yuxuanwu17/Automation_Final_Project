setwd("C://CMU/Courses/Automation/Project/")
load("C:/CMU/Courses/Automation/Project/5v_cleandf.RData")


## inspect data by checking NA ratio of each column
## remove features that more than or equal to 5% samples have a NA in this field
removeNAColumn <- function(data, threshold = 0.05) {
  NA_ratio <- c()
  for (i in 1:ncol(data)) {
    curr_NA_ratio <- sum(is.na(data[, i]))/nrow(data)
    NA_ratio <- c(NA_ratio, curr_NA_ratio)
  }
  cleaned_dat <- data[, which(NA_ratio < threshold)]
}

clean_dat <- removeNAColumn(df) # 584 feature + 1 label remaining

## remove samples has NA in any feature field
removeNARow <- function(data) {
  removeRowIndex <- c()
  for (i in 1:nrow(data)) {
    if (any(is.na(data[i, ]))) {
      removeRowIndex <- c(removeRowIndex, -i)
    }
  }
  newDat <- as.data.frame(data[removeRowIndex, ])
}

finalDat <- removeNARow(clean_dat)


## remove feature that are all the same between samples
## only one feature: “ecodesmachinery”
sameFeature <- c()
for (i in 1:ncol(finalDat)){
  if (length(table(finalDat[,i])) == 1) {
    sameFeature <- c(sameFeature, colnames(finalDat)[i])
  }
}
clean_dat <- finalDat[,which(colnames(finalDat)!= sameFeature[1])]

# write to file
write.csv(clean_dat, "Cleaned_dat.csv", row.names = FALSE)
