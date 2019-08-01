library(tidyverse)

parse_data <- function(base_filename,
                       datadir = "../refactor_Bayesian_CNN/results_for_SSCI2019/",
                       epochs = 100) {

  full_df <- data_frame()
  for (i in 0:5) {
    for (dataset in c("train", "val", "test")) {
      for (rand in c("CV", "VGMM-CV")) {
        if (rand == "CV") {
          filename <- paste0(datadir, base_filename,
                             i, "_", dataset, "_rand.txt")
        } else {
          filename <- paste0(datadir, base_filename,
                             i, "_", dataset, "_vgmm.txt")
        }

        if (file.exists(filename)) {
          print(paste0("Reading CV fold #",
                       i, ", ", dataset, " set, ",
                       "and rand = ", rand))
          cv_txt <- read_delim(filename, delim = "}", col_names = FALSE,
                               col_types = cols(.default = "c"))
        } else {
          warning(paste(filename, "does not exist!"))
          next
        }

        # There is a bug where the VGMM folds are numbered from 0 to 4 but
        # the random CV folds are numbered from 1 to 5. Taking care of that:
        if (rand == "CV") {
          cv_txt <- mutate(cv_txt, cv = i)
        } else {
          cv_txt <- mutate(cv_txt, cv = i + 1)
        }

        key <- list(train = "Train", val = "Validation", test = "Test")
        cv_df <- cv_txt %>%
          mutate(dataset = key[[dataset]], rand = rand) %>%
          gather(observation, value, -cv, -dataset, -rand) %>%
          drop_na()

        cv_df <- cv_df %>%
          mutate(epoch = gsub(".*Epoch': (\\d+).*", "\\1", value)) %>%
          mutate(accuracy = gsub(".*'Accuracy':\\s+tensor\\(([0-9]*\\.[0-9]+|[0-9]+).*",
                                 "\\1", value)) %>%
          mutate(loss = gsub(".*'Loss':\\s+tensor\\(([0-9]*\\.[0-9]+|[0-9]+).*",
                             "\\1", value)) %>%
          select(-value) %>%
          mutate(epoch = as.numeric(epoch), accuracy = as.numeric(accuracy),
                 loss = as.numeric(loss))

        if (!all(complete.cases(cv_df))) {
          stop(paste0("Missing values for CV #",
                      i, ", ", dataset, " set, ",
                      "and rand = ", rand))
        }

        if ( !all(sort(cv_df$epoch) == 1:epochs )) {
          stop(paste0("Unexpected number of repeted epochs for CV #",
                      i, ", ", dataset, " set, ",
                      "and rand = ", rand))
        }

        full_df <- bind_rows(full_df, cv_df)
      }
    }
  }

  return(full_df)
}


#--- Non-Bayesian 3conv3fc

# parse the data

NonBayes3conv3fc_mnist_cv_df <- parse_data("diagnostics_NonBayes3conv3fc_fashion-mnist_cv")
table(complete.cases(NonBayes3conv3fc_mnist_cv_df))

# plot accuracy

NonBayes3conv3fc_mnist_cv_df$dataset <- factor(NonBayes3conv3fc_mnist_cv_df$dataset,
                                              ordered = TRUE,
                                              levels = c("Train", "Validation", "Test"))
NonBayes3conv3fc_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, accuracy, color = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() +
    xlab("Epoch") + ylab("Accuracy")
  # + ggtitle("3conv3fc (non-Bayesian) on Fashion-MNIST")

ggsave("./output/NonBayes3conv3fc_fashion-mnist_cv_accuracy.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/NonBayes3conv3fc_fashion-mnist_cv_accuracy.pdf",
       width = 6.8, height = 3.3, units = "in")

# plot loss

NonBayes3conv3fc_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = Fold, linetype = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() + ylim(0, 10) +
    xlab("Epoch") + ylab("Loss")
  #+ ggtitle("3conv3fc (non-Bayesian) on Fashion-MNIST")

ggsave("./output/NonBayes3conv3fc_fashion-mnist_cv_loss.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/NonBayes3conv3fc_fashion-mnist_cv_loss.pdf",
       width = 6.8, height = 3.3, units = "in")

# mean and std of accuracy at epoch 100 for the VGMM splits

acc_df <- NonBayes3conv3fc_mnist_cv_df %>%
  filter(epoch == 100, rand == "VGMM-CV") %>%
  group_by(dataset) %>%
  summarize(mean_acc = mean(accuracy),
            std_acc = sqrt(var(accuracy))) %>%
  mutate_if(is.numeric, format, 1)
acc_df
# # A tibble: 3 x 3
#   dataset    mean_acc std_acc  
#   <ord>      <chr>    <chr>    
# 1 Train      97.97152 0.1251083
# 2 Validation 91.91050 0.2832904
# 3 Test       67.91618 5.1795364
paste(round(as.numeric(c(t(as.matrix(acc_df[,2:3])))), 2), collapse=" & ")


#--- Bayesian 3conv3fc

# parse the data

Bayes3conv3fc_mnist_cv_df <- parse_data("diagnostics_Bayes3conv3fc_fashion-mnist_cv")
table(complete.cases(Bayes3conv3fc_mnist_cv_df))

# plot accuracy

Bayes3conv3fc_mnist_cv_df$dataset <- factor(Bayes3conv3fc_mnist_cv_df$dataset,
                                           ordered = TRUE,
                                           levels = c("Train", "Validation", "Test"))
Bayes3conv3fc_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, accuracy, color = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() + xlab("Epoch") + ylab("Accuracy")
  #+ ggtitle("Bayesian 3conv3fc on Fashion-MNIST")

ggsave("./output/Bayes3conv3fc_fashion-mnist_cv_accuracy.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/Bayes3conv3fc_fashion-mnist_cv_accuracy.pdf",
       width = 6.8, height = 3.3, units = "in")

# plot loss

Bayes3conv3fc_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = Fold, shape = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) + ylim(0, 10) +
    theme_bw() + xlab("Epoch") + ylab("Loss")
  #+ ggtitle("Bayesian 3conv3fc on Fashion-MNIST")

ggsave("./output/Bayes3conv3fc_fashion-mnist_cv_loss.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/Bayes3conv3fc_fashion-mnist_cv_loss.pdf",
       width = 6.8, height = 3.3, units = "in")

# mean and std of accuracy at epoch 100 for the VGMM splits

acc_df <- Bayes3conv3fc_mnist_cv_df %>%
  filter(epoch == 100, rand == "VGMM-CV") %>%
  group_by(dataset) %>%
  summarize(mean_acc = mean(accuracy),
            std_acc = sqrt(var(accuracy))) %>%
  mutate_if(is.numeric, format, 1)
acc_df
# # A tibble: 3 x 3
#   dataset    mean_acc std_acc  
#   <ord>      <chr>    <chr>    
# 1 Train      98.68970 0.1945309
# 2 Validation 91.25644 0.4301867
# 3 Test       64.44000 4.1948470
paste(round(as.numeric(c(t(as.matrix(acc_df[,2:3])))), 2), collapse=" & ")


#--- Non-Bayesian AlexNet

# parse the data

NonBayesalexnet_mnist_cv_df <- parse_data("diagnostics_NonBayesalexnet_fashion-mnist_cv")
table(complete.cases(NonBayesalexnet_mnist_cv_df))

# plot accuracy

NonBayesalexnet_mnist_cv_df$dataset <- factor(NonBayesalexnet_mnist_cv_df$dataset,
                                              ordered = TRUE,
                                              levels = c("Train", "Validation", "Test"))
NonBayesalexnet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, accuracy, color = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() +
    xlab("Epoch") + ylab("Accuracy")
  # + ggtitle("AlexNet (non-Bayesian) on Fashion-MNIST")

ggsave("./output/NonBayesalexnet_fashion-mnist_cv_accuracy.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/NonBayesalexnet_fashion-mnist_cv_accuracy.pdf",
       width = 6.8, height = 3.3, units = "in")

# plot loss

NonBayesalexnet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = Fold, linetype = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() + ylim(0, 10) +
    xlab("Epoch") + ylab("Loss")
  #+ ggtitle("AlexNet (non-Bayesian) on Fashion-MNIST")

ggsave("./output/NonBayesalexnet_fashion-mnist_cv_loss.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/NonBayesalexnet_fashion-mnist_cv_loss.pdf",
       width = 6.8, height = 3.3, units = "in")

# mean and std of accuracy at epoch 100 for the VGMM splits

acc_df <- NonBayesalexnet_mnist_cv_df %>%
  filter(epoch == 100, rand == "VGMM-CV") %>%
  group_by(dataset) %>%
  summarize(mean_acc = mean(accuracy),
            std_acc = sqrt(var(accuracy))) %>%
  mutate_if(is.numeric, format, 1)
acc_df
# # A tibble: 3 x 3
#   dataset    mean_acc std_acc  
#   <ord>      <chr>    <chr>    
# 1 Train      98.97292 0.1343509
# 2 Validation 90.92658 0.4258816
# 3 Test       66.51252 4.8762226
paste(round(as.numeric(c(t(as.matrix(acc_df[,2:3])))), 2), collapse=" & ")


#--- Bayesian AlexNet

# parse the data

Bayesalexnet_mnist_cv_df <- parse_data("diagnostics_Bayesalexnet_fashion-mnist_cv")
table(complete.cases(Bayesalexnet_mnist_cv_df))

# plot accuracy

Bayesalexnet_mnist_cv_df$dataset <- factor(Bayesalexnet_mnist_cv_df$dataset,
                                           ordered = TRUE,
                                           levels = c("Train", "Validation", "Test"))
Bayesalexnet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, accuracy, color = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() + xlab("Epoch") + ylab("Accuracy")
  #+ ggtitle("Bayesian AlexNet on Fashion-MNIST")

ggsave("./output/Bayesalexnet_fashion-mnist_cv_accuracy.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/Bayesalexnet_fashion-mnist_cv_accuracy.pdf",
       width = 6.8, height = 3.3, units = "in")

# plot loss

Bayesalexnet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = Fold, shape = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) + ylim(0, 10) +
    theme_bw() + xlab("Epoch") + ylab("Loss")
  #+ ggtitle("Bayesian AlexNet on Fashion-MNIST")

ggsave("./output/Bayesalexnet_fashion-mnist_cv_loss.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/Bayesalexnet_fashion-mnist_cv_loss.pdf",
       width = 6.8, height = 3.3, units = "in")

# mean and std of accuracy at epoch 100 for the VGMM splits

acc_df <- Bayesalexnet_mnist_cv_df %>%
  filter(epoch == 100, rand == "VGMM-CV") %>%
  group_by(dataset) %>%
  summarize(mean_acc = mean(accuracy),
            std_acc = sqrt(var(accuracy))) %>%
  mutate_if(is.numeric, format, 1)
acc_df
# # A tibble: 3 x 3
#   dataset    mean_acc std_acc  
#   <ord>      <chr>    <chr>    
# 1 Train      96.33220 0.2873811
# 2 Validation 91.58410 0.2462084
# 3 Test       64.21422 5.6736701
paste(round(as.numeric(c(t(as.matrix(acc_df[,2:3])))), 2), collapse=" & ")


#--- Non-Bayesian LeNet

# parse the data

NonBayeslenet_mnist_cv_df <- parse_data("diagnostics_NonBayeslenet_fashion-mnist_cv")
table(complete.cases(NonBayeslenet_mnist_cv_df))

# plot accuracy

NonBayeslenet_mnist_cv_df$dataset <- factor(NonBayeslenet_mnist_cv_df$dataset,
                                            ordered = TRUE,
                                            levels = c("Train", "Validation", "Test"))
NonBayeslenet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, accuracy, color = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() +
    xlab("Epoch") + ylab("Accuracy")
  # + ggtitle("lenet (non-Bayesian) on Fashion-MNIST")

ggsave("./output/NonBayeslenet_fashion-mnist_cv_accuracy.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/NonBayeslenet_fashion-mnist_cv_accuracy.pdf",
       width = 6.8, height = 3.3, units = "in")

# plot loss

NonBayeslenet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = Fold, linetype = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() + ylim(0, 10) +
    xlab("Epoch") + ylab("Loss")
  #+ ggtitle("lenet (non-Bayesian) on Fashion-MNIST")

ggsave("./output/NonBayeslenet_fashion-mnist_cv_loss.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/NonBayeslenet_fashion-mnist_cv_loss.pdf",
       width = 6.8, height = 3.3, units = "in")

# mean and std of accuracy at epoch 100 for the VGMM splits

acc_df <- NonBayeslenet_mnist_cv_df %>%
  filter(epoch == 100, rand == "VGMM-CV") %>%
  group_by(dataset) %>%
  summarize(mean_acc = mean(accuracy),
            std_acc = sqrt(var(accuracy))) %>%
  mutate_if(is.numeric, format, 1)
acc_df
# # A tibble: 3 x 3
#   dataset    mean_acc std_acc  
#   <ord>      <chr>    <chr>    
# 1 Train      98.03414 0.2541528
# 2 Validation 91.44186 0.2937422
# 3 Test       65.43374 4.9976034
paste(round(as.numeric(c(t(as.matrix(acc_df[,2:3])))), 2), collapse=" & ")


#--- Bayesian lenet

# parse the data

Bayeslenet_mnist_cv_df <- parse_data("diagnostics_Bayeslenet_fashion-mnist_cv")
table(complete.cases(Bayeslenet_mnist_cv_df))

# plot accuracy

Bayeslenet_mnist_cv_df$dataset <- factor(Bayeslenet_mnist_cv_df$dataset,
                                           ordered = TRUE,
                                           levels = c("Train", "Validation", "Test"))
Bayeslenet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, accuracy, color = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() + xlab("Epoch") + ylab("Accuracy")
  #+ ggtitle("Bayesian lenet on Fashion-MNIST")

ggsave("./output/Bayeslenet_fashion-mnist_cv_accuracy.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/Bayeslenet_fashion-mnist_cv_accuracy.pdf",
       width = 6.8, height = 3.3, units = "in")

# plot loss

Bayeslenet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = Fold, shape = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) + ylim(0, 10) +
    theme_bw() + xlab("Epoch") + ylab("Loss")
  #+ ggtitle("Bayesian lenet on Fashion-MNIST")

ggsave("./output/Bayeslenet_fashion-mnist_cv_loss.png",
       width = 6.8, height = 3.3, units = "in")
ggsave("./output/Bayeslenet_fashion-mnist_cv_loss.pdf",
       width = 6.8, height = 3.3, units = "in")

# mean and std of accuracy at epoch 100 for the VGMM splits

acc_df <- Bayeslenet_mnist_cv_df %>%
  filter(epoch == 100, rand == "VGMM-CV") %>%
  group_by(dataset) %>%
  summarize(mean_acc = mean(accuracy),
            std_acc = sqrt(var(accuracy))) %>%
  mutate_if(is.numeric, format, 1)
acc_df
# # A tibble: 3 x 3
#   dataset    mean_acc std_acc  
#   <ord>      <chr>    <chr>    
# 1 Train      94.04024 0.2704761
# 2 Validation 90.54154 0.8211646
# 3 Test       63.18134 4.7217090
paste(round(as.numeric(c(t(as.matrix(acc_df[,2:3])))), 2), collapse=" & ")
