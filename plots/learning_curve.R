library(tidyverse)

parse_data <- function(base_filename, epochs = 30) {

  full_df <- data_frame()
  for (i in 0:5) {
    for (dataset in c("train", "val", "test")) {
      for (rand in c("CV", "VGMM-CV")) {
        if (rand == "CV") {
          filename <- paste0("./data/", base_filename,
                             i, "_", dataset, "_rand.txt")
        } else {
          filename <- paste0("./data/", base_filename,
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

        # dirty hacks:
        # Epoch 1 appears twice for the VGMM results for some reason.
        if (rand == "VGMM-CV") {
          ind1 <- which(cv_df$epoch == 1)
          if (length(ind1) > 1) {
            cv_df <- cv_df[ind1[length(ind1)]:nrow(cv_df), ]
          }
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


#--- Bayesian 3conv3fc

# TODO: this needs new data

#--- Non-Bayesian AlexNet

# parse the data

NonBayesalexnet_mnist_cv_df <- parse_data("diagnostics_NonBayesalexnet_mnist_cv",
                                          epochs = 100)
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

ggsave("./output/NonBayesalexnet_mnist_cv_accuracy.png")
ggsave("./output/NonBayesalexnet_mnist_cv_accuracy.pdf")

# plot loss

NonBayesalexnet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = Fold, linetype = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() +
    xlab("Epoch") + ylab("Loss")
  #+ ggtitle("AlexNet (non-Bayesian) on Fashion-MNIST")

ggsave("./output/NonBayesalexnet_mnist_cv_loss.png")
ggsave("./output/NonBayesalexnet_mnist_cv_loss.pdf")

# mean and std of accuracy at epoch 100 for the VGMM splits

NonBayesalexnet_mnist_cv_df %>%
  filter(epoch == 100, rand == "VGMM-CV") %>%
  group_by(dataset) %>%
  summarize(mean_acc = mean(accuracy),
            std_acc = sqrt(var(accuracy)))


#--- Bayesian AlexNet

# parse the data

Bayesalexnet_mnist_cv_df <- parse_data("diagnostics_Bayesalexnet_mnist_cv",
                                       epochs = 100)
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

ggsave("./output/Bayesalexnet_mnist_cv_accuracy.png")
ggsave("./output/Bayesalexnet_mnist_cv_accuracy.pdf")

# plot loss

Bayesalexnet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = Fold, shape = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() + xlab("Epoch") + ylab("Loss")
  #+ ggtitle("Bayesian AlexNet on Fashion-MNIST")

ggsave("./output/Bayesalexnet_mnist_cv_loss.png")
ggsave("./output/Bayesalexnet_mnist_cv_loss.pdf")

# mean and std of accuracy at epoch 100 for the VGMM splits

Bayesalexnet_mnist_cv_df %>%
  filter(epoch == 100, rand == "VGMM-CV") %>%
  group_by(dataset) %>%
  summarize(mean_acc = mean(accuracy),
            std_acc = sqrt(var(accuracy)))


#--- Non-Bayesian LeNet

# parse the data

NonBayeslenet_mnist_cv_df <- parse_data("diagnostics_NonBayeslenet_mnist_cv",
                                        epochs = 100)
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

ggsave("./output/NonBayeslenet_mnist_cv_accuracy.png")
ggsave("./output/NonBayeslenet_mnist_cv_accuracy.pdf")

# plot loss

NonBayeslenet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = Fold, linetype = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() +
    xlab("Epoch") + ylab("Loss")
  #+ ggtitle("lenet (non-Bayesian) on Fashion-MNIST")

ggsave("./output/NonBayeslenet_mnist_cv_loss.png")
ggsave("./output/NonBayeslenet_mnist_cv_loss.pdf")

# mean and std of accuracy at epoch 100 for the VGMM splits

NonBayeslenet_mnist_cv_df %>%
  filter(epoch == 100, rand == "VGMM-CV") %>%
  group_by(dataset) %>%
  summarize(mean_acc = mean(accuracy),
            std_acc = sqrt(var(accuracy)))


#--- Bayesian lenet

# parse the data

Bayeslenet_mnist_cv_df <- parse_data("diagnostics_Bayeslenet_mnist_cv",
                                       epochs = 100)
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

ggsave("./output/Bayeslenet_mnist_cv_accuracy.png")
ggsave("./output/Bayeslenet_mnist_cv_accuracy.pdf")

# plot loss

Bayeslenet_mnist_cv_df %>%
  mutate(Fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = Fold, shape = Fold)) +
    geom_line() +
    stat_summary(fun.y = mean, geom = "line", lwd = 0.7, aes(group = 1)) +
    facet_grid(rand~dataset) +
    theme_bw() + xlab("Epoch") + ylab("Loss")
  #+ ggtitle("Bayesian lenet on Fashion-MNIST")

ggsave("./output/Bayeslenet_mnist_cv_loss.png")
ggsave("./output/Bayeslenet_mnist_cv_loss.pdf")

# mean and std of accuracy at epoch 100 for the VGMM splits

Bayeslenet_mnist_cv_df %>%
  filter(epoch == 100, rand == "VGMM-CV") %>%
  group_by(dataset) %>%
  summarize(mean_acc = mean(accuracy),
            std_acc = sqrt(var(accuracy)))
