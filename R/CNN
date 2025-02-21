install.packages(c("tensorflow", "dplyr", "caret"))
library(tensorflow)
library(dplyr)
library(caret)
library(reticulate)

# Ensure TensorFlow is installed
install_tensorflow(version = "2.17.0")

# Load and preprocess data
data <- df
data$Sequence <- toupper(data$Sequence)  # Convert to uppercase
data$Sequence <- gsub("[^AGTC]", "", data$Sequence)  # Remove invalid characters
data$Sequence <- trimws(data$Sequence)

# Remove empty sequences
data <- data[nchar(data$Sequence) > 0 & !is.na(data$Sequence), ]

# Extract sequences and labels
sequences <- as.character(data$Sequence)
labels <- as.numeric(data$SMAR)

# Function to encode DNA sequences
encode_sequence <- function(seq) {
  char_map <- c("A" = 1, "G" = 2, "T" = 3, "C" = 4)
  seq_split <- unlist(strsplit(seq, ""))
  encoded <- sapply(seq_split, function(x) char_map[[x]], USE.NAMES = FALSE)
  encoded[is.na(encoded)] <- 0  # Replace NAs with 0
  return(encoded)
}

# Encode sequences
encoded_sequences <- lapply(sequences, encode_sequence)

# Pad sequences to uniform length
max_sequence_length <- max(sapply(encoded_sequences, length))
max_sequence_length <- as.integer(max_sequence_length)

padded_sequences <- do.call(rbind, lapply(encoded_sequences, function(seq) {
  length_diff <- max_sequence_length - length(seq)
  c(seq, rep(0, length_diff))  # Pad with zeros
}))

# Convert to matrix for TensorFlow
padded_sequences <- matrix(unlist(padded_sequences), nrow = length(encoded_sequences), byrow = TRUE)

# Split data into train and test sets
set.seed(42)
train_index <- createDataPartition(labels, p = 0.8, list = FALSE)

X_train <- padded_sequences[train_index, , drop = FALSE]
X_test  <- padded_sequences[-train_index, , drop = FALSE]
y_train <- labels[train_index]
y_test  <- labels[-train_index]

# Convert to numpy arrays for TensorFlow
numpy <- import("numpy")
X_train <- numpy$array(X_train, dtype = "int32")
y_train <- numpy$array(y_train, dtype = "int32")
X_test  <- numpy$array(X_test, dtype = "int32")
y_test  <- numpy$array(y_test, dtype = "int32")

# Define CNN Model using TensorFlow's Keras API
model <- tf$keras$Sequential(list(
  tf$keras$layers$Embedding(input_dim = 5L, output_dim = 128L),
  tf$keras$layers$Conv1D(filters = 64L, kernel_size = 5L, activation = "relu"),
  tf$keras$layers$MaxPooling1D(pool_size = 2L),
  tf$keras$layers$Dropout(rate = 0.3),
  tf$keras$layers$Conv1D(filters = 128L, kernel_size = 5L, activation = "relu"),
  tf$keras$layers$MaxPooling1D(pool_size = 2L),
  tf$keras$layers$Dropout(rate = 0.3),
  tf$keras$layers$Flatten(),
  tf$keras$layers$Dense(units = 128L, activation = "relu"),
  tf$keras$layers$Dropout(rate = 0.3),
  tf$keras$layers$Dense(units = 1L, activation = "sigmoid")
))

# Compile the model
model$compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list(tf$keras$metrics$BinaryAccuracy(name = "accuracy"))
)

# Train the model
batch_size <- 32L
epochs <- 10L

history <- model$fit(
  X_train, y_train,
  epochs = epochs,
  batch_size = batch_size,
  validation_split = 0.2,
  verbose = 1
)

# Evaluate the model
metrics <- model$evaluate(X_test, y_test, verbose = 0)
cat(sprintf("Test Loss: %.4f\n", metrics[[1]]))
cat(sprintf("Test Accuracy: %.4f\n", metrics[[2]]))

# Generate predictions
y_pred <- model$predict(X_test) > 0.5
y_pred <- as.numeric(y_pred)

# Confusion matrix
conf_matrix <- table(Predicted = y_pred, Actual = y_test)
print("Confusion Matrix:")
print(conf_matrix)
