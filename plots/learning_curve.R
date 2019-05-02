library(tidyverse)

parse_data <- function(base_filename, epochs = 30, folds = 5) {

  full_df <- data_frame()
  for (i in 0:folds) {
    for (train_or_test in c("train", "test")) {
      for (rand in c("CV", "VGMM-CV")) {
        if (rand == "CV") {
          filename <- paste0("./data/", base_filename,
                             i, "_", train_or_test, "_rand.txt")
        } else {
          filename <- paste0("./data/", base_filename,
                             i, "_", train_or_test, ".txt")
        }

        if (file.exists(filename)) {
          print(paste0("Reading CV fold #",
                       i, ", ", train_or_test, " set, ",
                       "and rand = ", rand))
          cv_txt <- read_delim(filename, delim = "}", col_names = FALSE,
                               col_types = cols(.default = "c"))
        } else {
          next
        }

        cv_df <- cv_txt %>%
          mutate(cv =  i, train_or_test = train_or_test, rand = rand) %>%
          gather(observation, value, -cv, -train_or_test, -rand) %>%
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
                      i, ", ", train_or_test, " set, ",
                      "and rand = ", rand))
        }

        # dirty hacks:
        # need only take the last set of epoch, which are indexed as 1,2,..,k in a
        # contiguous block of the dataframe rows, with k<30 for runs that didn't finish or k=30 for those that did.
        # TODO: Can (mostly?) be removed once all runs are completely finished, and the results are cleaned up.
        ind30 <- which(cv_df$epoch == epochs)
        if (rand == "VGMM-CV") {
          if (length(ind30) == 0) {
            # do nothing
          } else if (length(ind30) == 1) {
            if (ind30 + 1 < nrow(cv_df)) {
              cv_df <- cv_df[(ind30 + 1):nrow(cv_df), ]
            }
          } else if (length(ind30) == 2) {
            if (nrow(cv_df) == ind30[2]) {
              cv_df <- cv_df[(ind30[1] + 1):nrow(cv_df), ]
            } else {
              cv_df <- cv_df[(ind30[2] + 1):nrow(cv_df), ]
            }
          } else if (length(ind30) == 3) {
            cv_df <- cv_df[(ind30[1] + 1):nrow(cv_df), ]
          } else if (length(ind30) == 4) {
            cv_df <- cv_df[(ind30[2] + 1):(nrow(cv_df)-1), ]
          } else {
            stop(paste0("Unexpected number of repeted epochs for CV #",
                        i, ", ", train_or_test, " set, ",
                        "and rand = ", rand))
          }
        } else if (rand == "CV") {
          if (nrow(cv_df) >= epochs) {
            cv_df <- cv_df[1:epochs, ]
          }
        }

        if ( !( all(sort(cv_df$epoch) == 1:epochs) | length(cv_df) < epochs) ) {
          stop(paste0("Unexpected number of repeted epochs for CV #",
                      i, ", ", train_or_test, " set, ",
                      "and rand = ", rand))
        }

        full_df <- bind_rows(full_df, cv_df)
      }
    }
  }

  return(full_df)
}


#--- Bayesian 3conv3fc

# parse the data

Bayes3conv3fc_mnist_cv_df <- parse_data("diagnostics_Bayes3conv3fc_mnist_cv")
table(complete.cases(Bayes3conv3fc_mnist_cv_df))

# plot accuracy

Bayes3conv3fc_mnist_cv_df %>%
  mutate(fold = factor(cv)) %>%
  ggplot(aes(epoch, accuracy, color = fold, shape = fold, linetype = fold)) +
    geom_point() + geom_line() +
    facet_grid(rand~train_or_test) +
    theme_bw() + xlab("Epoch") + ylab("Accuracy")

ggsave("./output/Bayes3conv3fc_mnist_cv_accuracy.png")
ggsave("./output/Bayes3conv3fc_mnist_cv_accuracy.pdf")

# plot loss

Bayes3conv3fc_mnist_cv_df %>%
  mutate(fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = fold, shape = fold, linetype = fold)) +
    geom_point() + geom_line() +
    facet_grid(rand~train_or_test) +
    theme_bw() + xlab("Epoch") + ylab("Loss")

ggsave("./output/Bayes3conv3fc_mnist_cv_loss.png")
ggsave("./output/Bayes3conv3fc_mnist_cv_loss.pdf")


#--- Non-Bayesian AlexNet

# parse the data

NonBayesalexnet_mnist_cv_df <- parse_data("diagnostics_NonBayesalexnet_mnist_cv",
                                          epochs = 100)
table(complete.cases(Bayes3conv3fc_mnist_cv_df))

# plot accuracy

NonBayesalexnet_mnist_cv_df %>%
  mutate(fold = factor(cv)) %>%
  ggplot(aes(epoch, accuracy, color = fold, shape = fold)) +
    geom_line() +
    facet_grid(rand~train_or_test) +
    theme_bw() + xlab("Epoch") + ylab("Accuracy")

ggsave("./output/NonBayesalexnet_mnist_cv_accuracy.png")
ggsave("./output/NonBayesalexnet_mnist_cv_accuracy.pdf")

# plot loss

NonBayesalexnet_mnist_cv_df %>%
  mutate(fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = fold, shape = fold)) +
    geom_line() +
    facet_grid(rand~train_or_test) +
    theme_bw() + xlab("Epoch") + ylab("Loss")

ggsave("./output/NonBayesalexnet_mnist_cv_loss.png")
ggsave("./output/NonBayesalexnet_mnist_cv_loss.pdf")


#--- Bayesian AlexNet

# parse the data

Bayesalexnet_mnist_cv_df <- parse_data("diagnostics_Bayesalexnet_mnist_cv",
                                       epochs = 100)
table(complete.cases(Bayes3conv3fc_mnist_cv_df))

# plot accuracy

Bayesalexnet_mnist_cv_df %>%
  mutate(fold = factor(cv)) %>%
  ggplot(aes(epoch, accuracy, color = fold, shape = fold)) +
    geom_line() +
    facet_grid(rand~train_or_test) +
    theme_bw() + xlab("Epoch") + ylab("Accuracy")

ggsave("./output/Bayesalexnet_mnist_cv_accuracy.png")
ggsave("./output/Bayesalexnet_mnist_cv_accuracy.pdf")

# plot loss

Bayesalexnet_mnist_cv_df %>%
  mutate(fold = factor(cv)) %>%
  ggplot(aes(epoch, loss, color = fold, shape = fold)) +
    geom_line() +
    facet_grid(rand~train_or_test) +
    theme_bw() + xlab("Epoch") + ylab("Loss")

ggsave("./output/Bayesalexnet_mnist_cv_loss.png")
ggsave("./output/Bayesalexnet_mnist_cv_loss.pdf")
