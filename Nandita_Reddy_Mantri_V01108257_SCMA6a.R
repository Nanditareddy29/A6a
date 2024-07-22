install.packages(c("quantmod", "forecast", "ggplot2", "caret", "Metrics", "MLmetrics"))
install.packages("keras")
install.packages("tensorflow")
install.packages("auto.arima")

# Load necessary libraries
library(quantmod)
library(forecast)
library(ggplot2)
library(caret)
library(Metrics)
library(keras)
library(tensorflow)
library(MLmetrics)

# Define the ticker symbol for Ultratech
ticker_symbol <- "ULTRACEMCO.BO"

# Download the historical data
data <- getSymbols(ticker_symbol, src = "yahoo", from = "2021-04-01", to = "2024-03-31", auto.assign = FALSE)
data <- data[, "ULTRACEMCO.BO.Adjusted"]

# Display the first few rows of the data
head(data)

# Save the data to a CSV file
write.csv(data, "ULTCEM.csv")

# Check for missing values
cat("Missing values:\n")
print(sum(is.na(data)))

# Plot the data
ggplot(data = as.data.frame(data), aes(x = index(data), y = coredata(data))) +
  geom_line(color = "blue") +
  labs(title = "HUL.NS Adj Close Price", x = "Date", y = "Adj Close Price") +
  theme_minimal()

# Resample to monthly data
monthly_data <- to.monthly(data, indexAt = "lastof", OHLC = FALSE)

# Check if the data has at least two full periods (24 months for monthly data)
if (nrow(monthly_data) < 24) {
  stop("Not enough data to fit the Holt-Winters model. Need at least 24 months of data.")
}

# Convert monthly data to a time series object
monthly_ts <- ts(coredata(monthly_data), start = c(2021, 4), frequency = 12)

# Plot the decomposed components (for visualization purposes)
result <- decompose(monthly_ts, type = "multiplicative")
plot(result)

# Split the data into training and test sets
train_index <- 1:(length(monthly_ts) * 0.8)
train_data <- monthly_ts[train_index]
test_data <- monthly_ts[-train_index]

# Fit the Holt-Winters model
holt_winters_model <- HoltWinters(monthly_ts, seasonal = "additive")

# Forecast for the next year (12 months)
holt_winters_forecast <- forecast(holt_winters_model, h = 12)

# Plot the forecast
plot(holt_winters_forecast, main = "Holt-Winters Forecast", xlab = "Date", ylab = "Close Price")
lines(monthly_ts, col = "blue")
legend("topleft", legend = c("Observed", "Holt-Winters Forecast"), col = c("blue", "red"), lty = 1:2)

# Compute RMSE, MAE, MAPE, R-squared for Holt-Winters model
rmse <- RMSE(test_data, holt_winters_forecast$mean)
mae <- MAE(test_data, holt_winters_forecast$mean)
mape <- MAPE(test_data, holt_winters_forecast$mean)
r2 <- R2_Score(test_data, holt_winters_forecast$mean)
cat(sprintf('Holt-Winters - RMSE: %f, MAE: %f, MAPE: %f, R-squared: %f\n', rmse, mae, mape, r2))

# Fit auto_arima model
arima_model <- auto.arima(train_data)

# Print the model summary
summary(arima_model)

# Generate forecast
arima_forecast <- forecast(arima_model, h = length(test_data))

# Plot the original data, fitted values, and forecast
autoplot(arima_forecast) +
  autolayer(ts(test_data, start = start(arima_forecast$mean), frequency = frequency(arima_forecast$mean)), series = "Test Data") +
  theme_minimal()

# Compute RMSE, MAE, MAPE, R-squared for ARIMA model
rmse <- RMSE(test_data, arima_forecast$mean)
mae <- MAE(test_data, arima_forecast$mean)
mape <- MAPE(test_data, arima_forecast$mean)
r2 <- R2_Score(test_data, arima_forecast$mean)
cat(sprintf('ARIMA - RMSE: %f, MAE: %f, MAPE: %f, R-squared: %f\n', rmse, mae, mape, r2))

# Fit auto_arima model for daily data
arima_model_daily <- auto.arima(coredata(data), seasonal = TRUE, stepwise = TRUE)

# Print the model summary
summary(arima_model_daily)

# Generate forecast for the next 60 days
n_periods <- 60
arima_forecast_daily <- forecast(arima_model_daily, h = n_periods)

# Create future dates index
future_dates <- seq.Date(from = as.Date(index(data)[length(index(data))]), by = "day", length.out = n_periods)

# Plot the original data, fitted values, and forecast
autoplot(arima_forecast_daily) +
  autolayer(ts(coredata(data), start = start(arima_forecast_daily$mean), frequency = frequency(arima_forecast_daily$mean)), series = "Original Data") +
  theme_minimal()

# Initialize MinMaxScaler (using caret package)
scaler <- preProcess(data.frame(data), method = c("range"))

# Scale the data
scaled_data <- predict(scaler, data.frame(data))

# Function to create sequences
create_sequences <- function(data, target_col, sequence_length) {
  sequences <- list()
  labels <- list()
  for (i in 1:(nrow(data) - sequence_length)) {
    sequences[[i]] <- as.matrix(data[i:(i + sequence_length - 1), ])
    labels[[i]] <- data[i + sequence_length, target_col]
  }
  return(list(X = do.call(rbind, sequences), y = unlist(labels)))
}

# Define the target column index and sequence length
target_col <- 1  # Adj Close is the only column
sequence_length <- 30

# Create sequences
sequences <- create_sequences(scaled_data, target_col, sequence_length)
X <- array_reshape(sequences$X, dim = c(nrow(sequences$X), sequence_length, ncol(sequences$X)))
y <- sequences$y

# Split the data into training and testing sets (80% training, 20% testing)
train_size <- round(nrow(X) * 0.8)
X_train <- X[1:train_size, , ]
X_test <- X[(train_size + 1):nrow(X), , ]
y_train <- y[1:train_size]
y_test <- y[(train_size + 1):length(y)]

# Build the LSTM model
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(sequence_length, ncol(X))) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 50, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)

summary(model)

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error'
)

# Train the model
history <- model %>% fit(
  X_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(X_test, y_test),
  shuffle = FALSE
)

# Evaluate the model
loss <- model %>% evaluate(X_test, y_test)
cat(sprintf("Test Loss: %f\n", loss))

# Predict on the test set
y_pred <- model %>% predict(X_test)

# Inverse transform the predictions and true values to get them back to the original scale
y_test_scaled <- as.vector(predict(scaler, data.frame(y_test)))
y_pred_scaled <- as.vector(predict(scaler, data.frame(y_pred)))

# Compute RMSE, MAE, MAPE, R-squared for LSTM model
rmse <- RMSE(y_test_scaled, y_pred_scaled)
mae <- MAE(y_test_scaled, y_pred_scaled)
mape <- MAPE(y_test_scaled, y_pred_scaled)
r2 <- R2_Score(y_test_scaled, y_pred_scaled)
cat(sprintf('LSTM - RMSE: %f, MAE: %f, MAPE: %f, R-squared: %f\n', rmse, mae, mape, r2))
