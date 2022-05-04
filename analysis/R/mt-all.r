library(lmerTest)
library(lme4)
library("rstudioapi")
library(stringr)
library(ggplot2)
library(data.table)

setwd(dirname(getActiveDocumentContext()$path))
# Load Maptask measurements
mta <- read.csv('../../data/hcrc_maptask/all_models.csv')
unique(mta$model)
mt <- mta[mta$model == "srilm-LM_EN_swb",]
#  [1] rnn-ft-mp2-                                      rnn_0                                           
#  [3] rnn_1                                            rnn_2                                           
#  [9] rnn_7                                            gpt_0                                           
# [11] gpt_1                                            gpt_2                                           
# [17] gpt_7                                            gpt_eos_0                                       
# [19] gpt_eos_1                                        gpt_eos_2                                       
# [25] gpt_eos_7                                        gpt2-en-maptask-finetuned-context_full_sep_space
# [27] gpt2-en-maptask-finetuned-context_full_sep_eos   reference                                       
# [29] srilm-LM_EN_nopretrain                           srilm-LM_EN_pretrain                            
# [31] srilm-LM_EN_swb                                  srilm-LM_EN_pret_swb                            
# [33] srilm-LM_EN_wiki1pt                              gpt2-en-fin 
# to test: srilm-LM_EN_pret_swb, rnn-ft-mp2-, reference, gpt2-en-maptask-finetuned-context_full_sep_space

# Name variables
mt$logh <- log(mt$xu_h)
mt$logp <- log(mt$index)
mt$logt <- log(mt$theme_id)
mt$corpus <- "maptask"
#mt = mt[mt$sum_h != 0,] # srilm

# --------------- Position in dialogue ---------------
m <- lmer(logh ~ 1 + logp + (1 + logp | file), mt)
summary(m)

# --------------- Position in transaction ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt | file), mt)
summary(m)

# --------------- Position in transaction [follower] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|file), mt[mt$speaker == 'f',])
summary(m)

# --------------- Position in transaction [giver] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|file), mt[mt$speaker == 'g',])
summary(m)

######################################################################
#                         XU & REITTER                               #
######################################################################
# transform to data.table to plot
mt = data.table(mt)
mt$group = str_c(mt$corpus, ' ', mt$speaker)

m1 = lmer(normalised_h ~ theme_id + (1|theme_id) + (1|file), mt)
summary(m1) 


m1_1 = lmer(xu_h ~ theme_id + (1|theme_id) + (1|file), mt)
summary(m1_1)


# ent vs within-episode position, grouped by episode index
ps1 = ggplot(mt[(theme_id <= 10),], 
             aes(x = theme_id, y = normalised_h)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = corpus)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = corpus)) +
    #  facet_wrap(~label, nrow = 1) +
    scale_x_continuous(breaks = 1:10) +
    xlab('within-episode position') + ylab('entropy') +
    theme(legend.position = c(.85, .1), legend.direction = 'horizontal')
#pdf('e_vs_inPos_g.pdf', 9, 2.5)
#plot.new()
plot(ps1)
#dev.off()

ps2 = ggplot(mt[ (theme_id <= 10),], 
             aes(x = theme_id, y = xu_h)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = corpus)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = corpus)) +
    #  facet_wrap(~label, nrow = 1) +
    scale_x_continuous(breaks = 1:10) +
    xlab('within-episode position') + ylab('normalized entropy') +
    theme(legend.position = c(.85, .1), legend.direction = 'horizontal')
#pdf('ne_vs_inPos_g.pdf', 9, 2.5)
#plot.new()
plot(ps2)
#dev.off()

# MAPTASK initiator
m1 = lmer(normalised_h ~ theme_id + (1|theme_id) + (1|file), mt[speaker == 'g' & theme_id <= 10,])
summary(m1)


# MAPTASK responder
m5 = lmer(normalised_h ~ theme_id + (1|theme_id) + (1|file), mt[speaker == 'f' & theme_id <= 10,])
summary(m5)

p1 = ggplot(mt[(theme_id <= 10),], 
            aes(x = theme_id, y = xu_h)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = speaker)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = speaker)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = speaker)) +
    scale_x_continuous(breaks = 1:10) +
    #  theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy') 
#pdf('e_vs_inPos_role_new.pdf', 4, 4)
#plot.new()
plot(p1)
#dev.off()

# ------- xr like plots but with gf stats ---------
p1 = ggplot(mt[(theme_id <= 10),], 
            aes(x = logt, y = xu_h)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = speaker)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = speaker)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = speaker)) +
    scale_x_continuous(breaks = 1:10) +
    #  theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
#pdf('e_vs_inPos_role_new.pdf', 4, 4)
#plot.new()
plot(p1)
#dev.off()

# ------------- stationarity tests ----------------
# A stationary time series is one whose properties do not depend on the time at 
# which the series is observed. Thus, time series with trends, or with 
# seasonality, are not stationary — the trend and seasonality will affect the 
# value of the time series at different times.
library(fpp)
library(forecast)

mt = data.table(mt)
mt.test = mt[, {
  res1 = Box.test(xu_h)
  res2 = adf.test(xu_h)
  res3 = kpss.test(xu_h)
  res4 = pp.test(xu_h)
  .(boxpval = res1$p.value, adfpval = res2$p.value, kpsspval = res3$p.value, pppval = res4$p.value)
}, by = .(file, speaker)]

# how many series passed stationarity tests? - out of 39 files * 2 speakers = 78
nrow(mt.test[boxpval<.05,]) # 2 (2.5%) instead of 64, 25%
nrow(mt.test[adfpval<.05,]) # 59 (75.6%) instead of 211, 82.4%
nrow(mt.test[kpsspval>.05,]) # 70 (89.7%) instead of 245, 95.7%
nrow(mt.test[pppval<.05,]) # 78 instead of 256, 100%

# mt.new = mt[, {
#   .(index, xu_h, tsId = .GRP)
# }, by = .(file, speaker)]
# m = lmer(xu_h ~ index + (1|tsId), mt.new)
# summary(m)
# n.s.
# The stationarity property seems contradictory to the previous findings about 
# entropy increase in written text and spoken dialogue
# It indicates that the stationarity of entropy series does not conflict with the 
# entropy increasing trend predicted by the principle of ERC (Shannon, 1948). 
# We conjecture that stationarity satisfies because the effect size (Adj-R2) of 
# entropy increase is very small.
