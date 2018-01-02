kartmodels: Kart simulation training models
================

This project makes use of `kartsim` to explore multiple training models.

Capturing Data
--------------

We will use the `kartsim` package and `kartsim_capture()` to capture training and test data to train the models in this repo.

1.  Capture training data using:

``` r
library(kartsim)
kartsim_capture("capture/train")
```

1.  Followed by test data with:

``` r
kartsim_capture("capture/test")
```

Training using the CIFAR model
------------------------------

We will first try modeling this as an image classification problem using the [cifar10\_cnn](https://tensorflow.rstudio.com/keras/articles/examples/cifar10_cnn.html) Keras example.

``` r
tfruns::training_run("models/tf-cifar.R")
```

or in `cloudml` runnning:

``` r
cloudml::cloudml_train("models/tf-cifar.R")
```

use `tfdeploy` to validate that predictions over the trained model work by running:

``` r
tfdeploy::predict_savedmodel(list(array(0, c(32,32,3))))
```

    $predictions
                      output
    1 0.3322, 0.3316, 0.3363

Notice that `predict_savedmodel()` initializes a tensorflow session for each prediction, which takes too long:

``` r
system.time(
  tfdeploy::predict_savedmodel(list(array(0, c(32,32,3))))
)
```

       user  system elapsed 
      2.007   0.045   2.027 

Instead, we can preload the model and predict over a `graph` object as follows:

``` r
sess <- tensorflow::tf$Session()
graph <- tfdeploy::load_savedmodel(sess)

system.time(
  tfdeploy::predict_savedmodel(list(array(0, c(32,32,3))), graph, type = "graph", sess = sess)
)
```

       user  system elapsed 
      0.027   0.001   0.025

Which we can use to control the kart based on this model:

``` r
kartsim::kartsim_control(function(image, direction) {
  labels <- c("left", "forward", "right")
  input <- array(png::readPNG(image), c(32,32,3))
  result <- tfdeploy::predict_savedmodel(input, graph, type = "graph", sess = sess)
  scores <- result$predictions$activation[[1]]

  labels[which(scores == max(scores))]
})
```

    517/517 [==============================] - 54s 105ms/step - loss: 6.4093 - acc: 0.6013 - val_loss: 5.4995 - val_acc: 0.6588
    Epoch 2/2
    517/517 [==============================] - 55s 107ms/step - loss: 6.3691 - acc: 0.6048 - val_loss: 5.5059 - val_acc: 0.6577

Improving the CIFAR model with context
--------------------------------------

In order to improve accuracy, we can consider making the model "remember" the state of the previous direction to help give continuity while steering.

``` r
tfruns::training_run("models/tf-cifar-prev.R")
```

``` r
library(tensorflow)
sess <- tensorflow::tf$Session()
graph <- tfdeploy::load_savedmodel(sess, "savedmodel/")

tfdeploy::predict_savedmodel(
  list(
    list(
      input = array(0, c(32,32,3)),
      previous = array(c(0,1,0), c(3))
    )
  ),
  graph,
  type = "graph",
  sess = sess)
```

    $predictions
                activation_6
    1 0.0462, 0.9226, 0.0312

We can apply this control policy as follows:

``` r
previous <- c(0,0,0)
control_cifar_prev <- function(image, direction) {
  labels <- c("left", "forward", "right")
  input <- array(png::readPNG(image), c(32,32,3))
  
  result <- tfdeploy::predict_savedmodel(
    list(
      list(
        input = input,
        previous = array(previous, c(3))
      )
    ),
    graph, type = "graph", sess = sess)
  
  message(result)
  
  scores <- result$predictions$activation[[1]]
  direction <- which(scores == max(scores))
  
  previous <<- array(c(0,0,0), c(3))
  previous[direction] <<- 1

  labels[direction]
}

kartsim::kartsim_control(control_cifar_prev)
```
