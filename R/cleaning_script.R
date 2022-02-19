
# download and save data --------------------------------------------------

red <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")
white <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ";")
write.csv(red, "data/raw/red_wine.csv")
write.csv(white, "data/raw/white_wine.csv")

df <- rbind(red, white)

# install.packages("janitor")
df <- janitor::get_dupes(df)

# Preprocessing ------------------------------------------------------

imputeOutlier <- function(X, w=3){
  greater <- ifelse(X > mean(X)+(w*sd(X)), median(X)+(w*sd(X)), X)
  less <- ifelse(X < mean(X)-(w*sd(X)), median(X)-(w*sd(X)), X)
  X <- ifelse(X == greater, greater, less)
  return(X)
}

df[, 1:11] <- lapply(df[, 1:11], imputeOutlier)

scale <- function(x) (x-min(x))/(max(x) - min(x))
df[, 1:11] <- lapply(df[, 1:11], scale)

write.csv(df, "data/clean/wine_quality.csv")
