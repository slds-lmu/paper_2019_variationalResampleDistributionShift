library(tidyverse)

parse_data <- function(base_filename,
                       datadir = "../refactor_Bayesian_CNN/results_for_SSCI2019/",
                       epochs = 100) {
  filename <- paste0(datadir, base_filename, ".txt")
  cv_txt <- read_delim(filename, delim = "}", col_names = FALSE,
                       col_types = cols(.default = "c"))
  as.character(cv_txt[,4])

  results_df <- cv_txt %>%
    gather(observation, value) %>%
    drop_na() %>%
    mutate(epoch = gsub(".*Epoch': (\\d+).*", "\\1", value)) %>%
    mutate(accuracy = gsub(".*'Accuracy':\\s+tensor\\(([0-9]*\\.[0-9]+|[0-9]+).*", "\\1", value)) %>%
    mutate(loss = gsub(".*'Loss':\\s+tensor\\(([0-9]*\\.[0-9]+|[0-9]+).*",
                       "\\1", value)) %>%
    mutate(validation = grepl("Validation", value)) %>%
    select(-value) %>%
    mutate(epoch = as.numeric(epoch), accuracy = as.numeric(accuracy),
           loss = as.numeric(loss))

  return(results_df)
}


#--- Non-Bayesian 3conv3fc

NonBayes3conv3fc_df <- parse_data("diagnostics_NonBayes3conv3fc_fashion-mnist")
table(complete.cases(NonBayes3conv3fc_df))
NonBayes3conv3fc_df %>%
  ggplot(aes(epoch, accuracy, color=validation)) +
    geom_line()
NonBayes3conv3fc_df %>% filter(epoch == 100) %>% .$accuracy
# [1] 97.37 91.08

#--- Bayesian 3conv3fc

Bayes3conv3fc_df <- parse_data("diagnostics_Bayes3conv3fc_fashion-mnist")
table(complete.cases(Bayes3conv3fc_df))
Bayes3conv3fc_df %>%
  ggplot(aes(epoch, accuracy, color=validation)) +
    geom_line()
Bayes3conv3fc_df %>% filter(epoch == 100) %>% .$accuracy
# [1] 98.7413 90.5760


#--- Non-Bayesian AlexNet

# parse the data

NonBayesalexnet_df <- parse_data("diagnostics_NonBayesalexnet_fashion-mnist")
table(complete.cases(NonBayesalexnet_df))
NonBayesalexnet_df %>%
  ggplot(aes(epoch, accuracy, color=validation)) +
    geom_line()
NonBayesalexnet_df %>% filter(epoch == 100) %>% .$accuracy
# [1] 98.7783 89.4200


#--- Bayesian AlexNet

Bayesalexnet_df <- parse_data("diagnostics_Bayesalexnet_fashion-mnist")
table(complete.cases(Bayesalexnet_df))
Bayesalexnet_df %>%
  ggplot(aes(epoch, accuracy, color=validation)) +
    geom_line()
Bayesalexnet_df %>% filter(epoch == 100) %>% .$accuracy
# [1] 95.8608 90.5620


#--- Non-Bayesian LeNet

NonBayeslenet_df <- parse_data("diagnostics_NonBayeslenet_fashion-mnist")
table(complete.cases(NonBayeslenet_df))
NonBayeslenet_df %>%
  ggplot(aes(epoch, accuracy, color=validation)) +
    geom_line()
NonBayeslenet_df %>% filter(epoch == 100) %>% .$accuracy
# [1] 97.705 89.770


#--- Bayesian lenet

Bayeslenet_df <- parse_data("diagnostics_Bayeslenet_fashion-mnist")
table(complete.cases(Bayeslenet_df))
Bayeslenet_df %>%
  ggplot(aes(epoch, accuracy, color=validation)) +
    geom_line()
Bayeslenet_df %>% filter(epoch == 100) %>% .$accuracy
# [1] 93.3772 89.0630
