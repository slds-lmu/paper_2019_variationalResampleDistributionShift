library(tidyverse)

results_cluster <- read_csv("./data/results_cluster.txt")
results_random <- read_csv("./data/results_random.txt")
results_cluster <- mutate(results_cluster, CV = "VGMM")
results_random <- mutate(results_random, CV = "Random")
results <- bind_rows(results_cluster, results_random)

results %>%
  ggplot(aes(x = CV, y = accuracy)) + geom_boxplot() +
    theme_bw() + theme(axis.title.x = element_blank()) +
    ylab("Accuracy")

ggsave("./output/accuracy_boxplot.png")
ggsave("./output/accuracy_boxplot.pdf")
