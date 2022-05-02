library(lmerTest)
library(lme4)
library("rstudioapi")
library(stringr)
library(ggplot2)
library(data.table)

setwd(dirname(getActiveDocumentContext()$path))
# Load Dataset measurements
#pc <- read.csv('../../data/paco-cheese/gpt-finedtuned-paco-cheese-c5.csv')
#pc <- read.csv('../../data/paco-cheese/gpt2-fr-training_file_1024.csv')
#pc <- read.csv('../../data/paco-cheese/gpt2-fr-paco-cheese-finetuned-paco-cheese-cs.csv')
#pc <- read.csv('../../data/paco-cheese/gpt2-fr-paco-cheese-finetuned-paco-cheese-eos-me.csv')
pc <- read.csv('../../data/paco-cheese/gpt2-fr-paco-cheese-finetuned-context_ 000—bs8.csv')
#pc <- pc[pc$model == ''] # in case of file containing results for several models
# Columns: 
# corpus,file,dyad,index,speaker,start,stop,text,theme,theme_role,theme_index,has_theme,
# context_5,con+text,normalised_h,length,tokens_h,sum_h,xu_h

# Adapt from Python
pc$index = pc$index + 1
pc$theme_index = pc$theme_index + 1
# Name variables
pc$logh <- log(pc$xu_h)
pc$logp <- log(pc$index)
pc$logt <- log(pc$theme_index)

# --------------- Position in dialogue ---------------
m <- lmer(logh ~ 1 + logp + (1 + logp | file), pc)
summary(m)


# --------------- Position in transaction ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt | file), pc)
summary(m)

# --------------- Position in transaction [follower] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|file), pc[pc$theme_role == 'f',])
summary(m)


# --------------- Position in transaction [giver] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|file), pc[pc$theme_role == 'g',])
summary(m)



######################################################################
#                         XU & REITTER                               #
######################################################################
# transform to data.table to plot
pc = data.table(pc)
pc$group = str_c(pc$corpus, ' ', pc$theme_role)

m1 = lmer(normalised_h ~ theme_index + (1|theme) + (1|file), pc)
summary(m1) 


m1_1 = lmer(xu_h ~ theme_index + (1|theme) + (1|file), pc)
summary(m1_1)


# ent vs within-episode position, grouped by episode index
ps1 = ggplot(pc[(!theme == "") & (theme != 'transition') & (theme_index <= 10),], 
             aes(x = theme_index, y = normalised_h)) +
  stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = corpus)) +
  stat_summary(fun.y = mean, geom = 'line', aes(lty = corpus)) +
#  facet_wrap(~label, nrow = 1) +
  scale_x_continuous(breaks = 1:10) +
  xlab('within-episode position') + ylab('entropy') +
  theme(legend.position = c(.85, .1), legend.direction = 'horizontal')
#pdf('e_vs_inPos_g.pdf', 9, 2.5)
plot(ps1)
#dev.off()

ps2 = ggplot(pc[(!theme == "") & (theme != 'transition') & (theme_index <= 10),], 
             aes(x = theme_index, y = xu_h)) +
  stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = corpus)) +
  stat_summary(fun.y = mean, geom = 'line', aes(lty = corpus)) +
#  facet_wrap(~label, nrow = 1) +
  scale_x_continuous(breaks = 1:10) +
  xlab('within-episode position') + ylab('normalized entropy') +
  theme(legend.position = c(.85, .1), legend.direction = 'horizontal')
#pdf('ne_vs_inPos_g.pdf', 9, 2.5)
plot(ps2)
#dev.off()

# PACO-CHEESE initiator
m1 = lmer(normalised_h ~ theme_index + (1|theme) + (1|file), pc[theme_role == 'g' & theme_index <= 10,])
summary(m1)


# PACO-CHEESE responder
m5 = lmer(normalised_h ~ theme_index + (1|theme) + (1|file), pc[theme_role == 'f' & theme_index <= 10,])
summary(m5)


pc$group = str_c(pc$corpus, ': ', pc$theme_role)
gg_color_hue <- function(n) {
  hues = seq(15, 375, length=n+1)
  hcl(h=hues, l=65, c=100)[1:n]
}
my_colors = gg_color_hue(2)

p1 = ggplot(pc[(theme != "") & (theme != 'transition') & (theme_index <= 10),], 
            aes(x = theme_index, y = xu_h)) +
  stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
  stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
  stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
  scale_x_continuous(breaks = 1:10) +
#  theme(legend.position = c(.75, .2)) +
  xlab('within-episode position') + ylab('entropy') +
  scale_fill_manual(values = c('cheese: g' = my_colors[1], 'cheese: f' = my_colors[1],
                               'paco: g' = my_colors[2], 'paco: f' = my_colors[2])) +
  scale_linetype_manual(values = c('cheese: g' = 1, 'cheese: f' = 3, 'paco: g' = 1, 'paco: f' = 3)) +
  scale_shape_manual(values = c('cheese: g' = 1, 'cheese: f' = 1, 'paco: g' = 4, 'paco: f' = 4))
#pdf('e_vs_inPos_role_new.pdf', 4, 4)
plot(p1)
#dev.off()

# ------- xr like plots but with gf stats ---------
p1 = ggplot(pc[(theme != "") & (theme != 'transition') & (theme_index <= 10),], 
            aes(x = logt, y = xu_h)) +
  stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
  stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
  stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
  scale_x_continuous(breaks = 1:10) +
  #  theme(legend.position = c(.75, .2)) +
  xlab('within-episode position') + ylab('entropy') +
  scale_fill_manual(values = c('cheese: g' = my_colors[1], 'cheese: f' = my_colors[1],
                               'paco: g' = my_colors[2], 'paco: f' = my_colors[2])) +
  scale_linetype_manual(values = c('cheese: g' = 1, 'cheese: f' = 3, 'paco: g' = 1, 'paco: f' = 3)) +
  scale_shape_manual(values = c('cheese: g' = 1, 'cheese: f' = 1, 'paco: g' = 4, 'paco: f' = 4))
#pdf('e_vs_inPos_role_new.pdf', 4, 4)
plot(p1)