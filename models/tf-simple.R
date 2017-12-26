library(keras)
library(tfruns)

model <- keras_model_sequential()

model <- model %>%
  layer_dense(
    1024,
    input_shape = c(32, 32, 3)) %>%
  layer_flatten() %>%
  layer_dense(3) %>%
  layer_activation("softmax")

opt <- optimizer_sgd()

model %>% compile(
  loss = "mse",
  optimizer = opt,
  metrics = "accuracy"
)

library(png)
classes <- c("left", "forward", "right")

batch_size <- 32

prepare_flow_images <- function(source_path) {
  output_path <- tempfile()
  dir.create(output_path)
  for (path in dir(source_path, full.names = T)) {
    for (d in c("left", "forward", "right")) {
      if (grepl(d, path)) {
        if (!file.exists(file.path(output_path, d))) dir.create(file.path(output_path, d))
        file.copy(path, file.path(output_path, d, basename(path)))
      }
    }
  }
  output_path
}

train_path <- prepare_flow_images("capture/train")
test_path <- prepare_flow_images("capture/test")

model %>% fit_generator(
  flow_images_from_directory(
    train_path,
    classes = classes,
    batch_size = batch_size,
    target_size = c(32, 32)),
  steps_per_epoch = as.integer(length(dir(train_path, recursive = T)) / batch_size), 
  epochs = 5,
  validation_data = flow_images_from_directory(
    test_path,
    classes = classes,
    batch_size = batch_size,
    target_size = c(32, 32))
)

model %>% export_savedmodel("savedmodel")