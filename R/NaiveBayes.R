if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("Biostrings")
if (!require(caret)) install.packages("caret")

library(Biostrings)
library(e1071)
library(tidyverse)
library(caret)

#### Function to compute k-mer counts for a sequence
compute_kmer_counts <- function(sequence, k = 4) {
  # Convert sequence to a DNAString object
  dna_seq <- DNAString(sequence)

  # Generate all possible k-mers for given k
  kmers <- oligonucleotideFrequency(dna_seq, width = k)

  # Convert named vector to data frame
  as.data.frame(as.list(kmers))
}

# Apply k-mer counting to each sequence
df$Sequence <- gsub("[^ATGC]", "", df$Sequence)
k <- 4  # Set the desired k-mer length
kmer_features <- do.call(rbind, lapply(df$Sequence, compute_kmer_counts, k = k))

# Combine k-mer features with the target column
data_prepared <- cbind(kmer_features, Target = df$SMAR)



#### Split the data into training and test sets
set.seed(123)
train_index <- sample(1:nrow(data_prepared), 0.8 * nrow(data_prepared))
train_data <- data_prepared[train_index, ]
test_data <- data_prepared[-train_index, ]

# Train the Naive Bayes model
nb_model <- naiveBayes(Target ~ ., data = train_data)

# Make predictions on the test set
nb_pred <- predict(nb_model, test_data)


nb_confusion_matrix <- table(Predicted = nb_pred, Actual = test_data$Target)
print(nb_confusion_matrix)

accuracy <- mean(nb_pred == test_data$Target)
cat("Model Accuracy:", accuracy, "\n")


test_data$Target <- factor(test_data$SMAR)
predictions <- factor(nb_pred, levels = levels(test_data$SMAR))

# Now, generate the confusion matrix
nb_confusion <- confusionMatrix(nb_pred, test_data$SMAR)

# Display confusion matrix and metrics
print(nb_confusion)


#stats
# Extract specific metrics
accuracy <- nb_confusion$overall['Accuracy']
precision <- nb_confusion$byClass['Pos Pred Value']  # This is Precision
recall <- nb_confusion$byClass['Sensitivity']        # This is Recall
f1_score <- (2 * precision * recall) / (precision + recall)  # F1 Score

cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
