library(lmerTest)
library(lme4)
library("rstudioapi")
library(stringr)
library(ggplot2)
library(data.table)

setwd(dirname(getActiveDocumentContext()$path))
# Load Dataset measurements
pca <- read.csv('../../data/paco-cheese/all_models.csv')
unique(pca$model)
pc <- pca[pca$model == "srilm-LM_FR_orfeocid",]
# [1] "rnn-ft-pc2-"                             [18] "gpt2-fr-eos-orfeo-cid-paco-cheese-context_full_sep_space"
# [2] "rnn_0"                                   [19] "gpt2-fr-eos-orfeo-cid-paco-cheese-context_full_sep_eos"  
# [3] "rnn_1"                                   [20] "gpt_space_0"                                             
# [4] "rnn_2"                                   [21] "gpt_space_1"                                             
# [9] "rnn_7"                                   [26] "gpt_space_6"                                             
# [10] "gpt_eos_0"                               [27] "gpt_space_7"                                             
# [11] "gpt_eos_1"                               [28] "gpt2-fr-paco-cheese-finetuned-context_full_sep_space"    
# [12] "gpt_eos_2"                               [29] "gpt2-fr-paco-cheese-finetuned-context_full_sep_eos"      
# [13] "gpt_eos_3"                               [30] "gpt2-fr-fin"                                             
# [14] "gpt_eos_4"                               [31] "srilm-LM_FR_pretrain"                                    
# [15] "gpt_eos_5"                               [32] "srilm-LM_FR_pret_orfeocid"                               
# [16] "gpt_eos_6"                               [33] "srilm-LM_FR_nopretrain"                                  
# [17] "gpt_eos_7"                               [34] "srilm-LM_FR_orfeocid" 
# to test: srilm-LM_FR_orfeocid, gpt2-fr-eos-orfeo-cid-paco-cheese-context_full_sep_eos

# Adapt from Python
pc$index = pc$index + 1
pc$theme_index = pc$theme_index + 1
# Name variables
pc$logh <- log(pc$xu_h)
pc$logp <- log(pc$index)
pc$logt <- log(pc$theme_index)
dim(pc[pc$sum_h == 0,]) 
pc = pc[pc$sum_h != 0,] # might not be needed for other than srilm

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


# ------------- stationarity tests ----------------
# A stationary time series is one whose properties do not depend on the time at 
# which the series is observed. Thus, time series with trends, or with 
# seasonality, are not stationary — the trend and seasonality will affect the 
# value of the time series at different times.
library(fpp)
library(forecast)

pc = data.table(pc)
pc.test = pc[, {
  res1 = Box.test(xu_h)
  res2 = adf.test(xu_h)
  res3 = kpss.test(xu_h)
  res4 = pp.test(xu_h)
  .(boxpval = res1$p.value, adfpval = res2$p.value, kpsspval = res3$p.value, pppval = res4$p.value)
}, by = .(file, speaker)]

# how many series passed stationarity tests? - out of 39 files * 2 speakers = 78
nrow(pc.test[boxpval<.05,]) # 2 (2.5%) instead of 64, 25%
nrow(pc.test[adfpval<.05,]) # 59 (75.6%) instead of 211, 82.4%
nrow(pc.test[kpsspval>.05,]) # 70 (89.7%) instead of 245, 95.7%
nrow(pc.test[pppval<.05,]) # 78 instead of 256, 100%

# pc.new = pc[, {
#   .(index, xu_h, tsId = .GRP)
# }, by = .(file, speaker)]
# m = lmer(xu_h ~ index + (1|tsId), pc.new)
# summary(m)
# n.s.
# The stationarity property seems contradictory to the previous findings about 
# entropy increase in written text and spoken dialogue
# It indicates that the stationarity of entropy series does not conflict with the 
# entropy increasing trend predicted by the principle of ERC (Shannon, 1948). 
# We conjecture that stationarity satisfies because the effect size (Adj-R2) of 
# entropy increase is very small.
