# plot for 2:8 training mode
# read data
dat <- read.csv("2-8.csv", header = TRUE)

plot_dat <- data.frame(name = c("Logistic regression", "Random forest", 
                                "XGBoost", 'KNN'),
                       mean = apply(dat, 2, mean),
                       sd = apply(dat, 2, sd))

library(ggplot2)

# Most basic error bar
p <- ggplot(plot_dat) +
  geom_bar(aes(x=name, y=mean), stat="identity", fill="skyblue", alpha=1) +
  geom_errorbar(aes(x=name, ymin=mean-sd, ymax=mean+sd), width=0.4, colour="orange", 
                alpha=0.9, size=1)  + coord_cartesian(ylim=c(0.75,0.85)) + 
  theme_bw()

pdf("D://p.pdf", height = 6, width = 6)
print(p)
dev.off()
