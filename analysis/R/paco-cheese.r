library(lmerTest)
library(lme4)

# Load Maptask measurements
pc <- read.csv('../../data/paco-cheese/pc_gpt1_c_2.csv')
# Columns: 
# corpus,file,dyad,index,speaker,start,stop,text,theme,theme_role,theme_index,has_theme,
# context_5,con+text,normalised_h,length,tokens_h,sum_h,xu_h


# Name variables
pc$logh <- log(pc$xu_h)
pc$logp <- log(pc$index + 1)
pc$logt <- log(pc$theme_index + 1)

# --------------- Position in dialogue ---------------
m <- lmer(logh ~ 1 + logp + (1 + logp | file), pc)
summary(m)
# REML criterion at convergence: 284
# 
# Scaled residuals: 
#     Min      1Q  Median      3Q     Max 
# -7.7381 -0.4073  0.0898  0.5598  4.3822 
# 
# Random effects:
#     Groups   Name        Variance  Std.Dev. Corr 
# file     (Intercept) 1.402e-04 0.011839      
# logp        3.723e-05 0.006102 -1.00
# Residual             6.265e-02 0.250294      
# Number of obs: 3816, groups:  file, 7
# 
# Fixed effects:
#     Estimate Std. Error        df t value Pr(>|t|)    
# (Intercept)  0.253297   0.022837 58.611383   11.09 5.04e-16 ***
#     logp        -0.053715   0.004755 10.877155  -11.30 2.40e-07 ***
#     ---
#     Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Correlation of Fixed Effects:
#     (Intr)
# logp -0.938
# convergence code: 0
# Model failed to converge with max|grad| = 0.0203197 (tol = 0.002, component 1)


# --------------- Position in transaction ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt | file), pc)
summary(m)
# REML criterion at convergence: 438.1
# 
# Scaled residuals: 
#     Min      1Q  Median      3Q     Max 
# -7.2583 -0.4186  0.0571  0.5377  4.1724 
# 
# Random effects:
#     Groups   Name        Variance  Std.Dev. Corr 
# file     (Intercept) 2.734e-04 0.016536      
# logt        8.652e-05 0.009302 -0.79
# Residual             6.523e-02 0.255410      
# Number of obs: 3816, groups:  file, 7
# 
# Fixed effects:
#     Estimate Std. Error        df t value Pr(>|t|)
# (Intercept) -0.009028   0.011319  5.367167  -0.798    0.459
# logt        -0.009069   0.004962  5.597111  -1.828    0.121
# 
# Correlation of Fixed Effects:
#     (Intr)
# logt -0.838


# --------------- Position in transaction [follower] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|file), pc[pc$theme_role == 'f',])
summary(m)
# REML criterion at convergence: 76
# 
# Scaled residuals: 
#     Min      1Q  Median      3Q     Max 
# -7.4553 -0.3795  0.0580  0.5315  3.5621 
# 
# Random effects:
#     Groups   Name        Variance  Std.Dev. Corr
# file     (Intercept) 5.547e-05 0.007448     
# logt        6.718e-06 0.002592 1.00
# Residual             6.096e-02 0.246897     
# Number of obs: 1448, groups:  file, 7
# 
# Fixed effects:
#     Estimate Std. Error        df t value Pr(>|t|)
# (Intercept) -0.018368   0.021360 57.153560  -0.860    0.393
# logt        -0.008930   0.007196 55.579549  -1.241    0.220
# 
# Correlation of Fixed Effects:
#     (Intr)
# logt -0.916
# convergence code: 0
# boundary (singular) fit: see ?isSingular


# --------------- Position in transaction [giver] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|file), pc[pc$theme_role == 'g',])
summary(m)

# REML criterion at convergence: 324.6
# 
# Scaled residuals: 
#     Min      1Q  Median      3Q     Max 
# -4.9217 -0.4428  0.0496  0.5439  4.0418 
# 
# Random effects:
#     Groups   Name        Variance  Std.Dev. Corr 
# file     (Intercept) 0.0018409 0.04291       
# logt        0.0002109 0.01452  -0.93
# Residual             0.0684406 0.26161       
# Number of obs: 1928, groups:  file, 7
# 
# Fixed effects:
#     Estimate Std. Error        df t value Pr(>|t|)  
# (Intercept)  0.040214   0.022011  5.840639   1.827   0.1188  
# logt        -0.021824   0.007696  6.247479  -2.836   0.0284 *
#     ---
#     Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Correlation of Fixed Effects:
#     (Intr)
# logt -0.920

