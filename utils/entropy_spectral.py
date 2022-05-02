# %% Adapted from Xu & Reitter 2017, acl2017-spectral-master
import pandas as pd
import numpy as np
import os,sys,re
from glob import glob

from scipy import stats, signal, fft
from statsmodels.regression.linear_model import yule_walker
from functools import partial

from scipy.interpolate import interp1d

# %%
# python spectrum R lib
SPEC_PATH = "/Users/neako/Documents/Cours-MasCo/PhD/tools/githubs/IT/time-series-analysis-master/Python"
sys.path.append(SPEC_PATH)
from spectrum import * # spec_pgram

# %%
# load clean data for comparison
df = pd.read_csv("/Users/neako/Downloads/xr_ent_swbd.csv", index_col=0)
df.rename(columns={'observation':'file', 'utterID':'index', 'ent_swbd':'xu_h'}, inplace=True)
df = df.sort_values(['file','index'])
df.head()

# %% [markdown]
# **PSO**
# ```R
# dt.ent_swbd = dt[, {
#         if (.N > 0) {
#             specval = spec.pgram(ent_swbd, taper=0, log='no', plot=FALSE)
#             .(spec = specval$spec, freq = specval$freq)
#         }
#     }, by = .(observation, who)]
# 
# ##
# # Compute the area under curve (AUV) for spectral plots
# # and then compute the ratio of the common area (power spectral overlap, PSO)
# dt.ent_swbd_pso = dt.ent_swbd[, {
#         x_g = freq[who=='g']
#         y_g = spec[who=='g']
#         x_f = freq[who=='f']
#         y_f = spec[who=='f']
#         # linear interpolation
#         x_out = sort(union(x_g, x_f)) #### concat + set (unique values)
#         approx_g = approx(x_g, y_g, xout = x_out)
#         approx_f = approx(x_f, y_f, xout = x_out)
#         # find min ys and remove NAs
#         x_out_g = x_out[which(!is.na(approx_g$y))]
#         y_out_g = approx_g$y[which(!is.na(approx_g$y))]
#         x_out_f = x_out[which(!is.na(approx_f$y))]
#         y_out_f = approx_f$y[which(!is.na(approx_f$y))]
#         y_min = pmin(approx_g$y, approx_f$y)
#         x_min = x_out[which(!is.na(y_min))]
#         y_min = y_min[which(!is.na(y_min))]
#         y_max = pmax(approx_g$y, approx_f$y)
#         x_max = x_out[!is.na(y_max)]
#         y_max = y_max[!is.na(y_max)]
#         # compute AUVs and PSO
#         AUV_g = trapz(x_out_g, y_out_g)
#         AUV_f = trapz(x_out_f, y_out_f)
#         AUV_min = trapz(x_min, y_min)
#         AUV_max = trapz(x_max, y_max)
#         # PSO = AUV_min / (AUV_g + AUV_f)
#         PSO = AUV_min / AUV_max
#         # return PSO
#         .(PSO = PSO, AUVg = AUV_g, AUVf = AUV_f, AUVmin = AUV_min)
#     }, by = observation]
# ```

# %%
spgram = partial(spec_pgram, taper=0, plot=False, log='no')
df_ent = df.groupby(['file','who']).agg({'xu_h':spgram})
df_ent_pso = df_ent.xu_h.apply(pd.Series)[['freq','spec']].explode(['freq','spec']).reset_index(drop=False)
df_ent_pso.freq = df_ent_pso.freq.astype(float)
df_ent_pso.spec = df_ent_pso.spec.astype(float)
df_ent_pso # shape OK

# %%
pso = []
for f in df.file.unique():
    df_ent_sel = df_ent_pso[df_ent.file == f]
    x_g = df_ent_sel[df_ent_sel.who == 'g'].freq
    y_g = df_ent_sel[df_ent_sel.who == 'g'].spec
    x_f = df_ent_sel[df_ent_sel.who == 'f'].freq
    y_f = df_ent_sel[df_ent_sel.who == 'f'].spec
    # linear interpolation - slightly modified to accompodate python interp1d which throws error instead of NA for values not in x
    x_out = pd.concat([x_g, x_f]).astype(float).sort_values().unique()
    x_out_g = x_out[np.where((x_out >= x_g.min()) & (x_out <= x_g.max()))[0]]
    x_out_f = x_out[np.where((x_out >= x_f.min()) & (x_out <= x_f.max()))[0]]
    # CANNOT USE THOSE DIRECTLY since shape differ
    y_out_f = interp1d(x_f, y_f)(x_out_f) # aka approx_[f,g]
    y_out_g = interp1d(x_g, y_g)(x_out_g) # approx$y est la valeur de np.intrep
    # ADDING NANs
    approx_f = interp1d(x_f, y_f) 
    approx_g = interp1d(x_g, y_g)
    approx_f = np.array([approx_f(x) if x in x_out_f else np.nan for x in x_out]) # contains out values
    approx_g = np.array([approx_g(x) if x in x_out_g else np.nan for x in x_out]) # contains out values
    # NOTE: np.interp(x_out, x_f, y_f) also works but doesn't add nans, adds 1rst non nan value in place
    
    # find min ys and remove NAs
    y_min = np.minimum(approx_f, approx_g)
    x_min = x_out[~np.isnan(y_min)]
    y_min = y_min[~np.isnan(y_min)]
    y_max = np.maximum(approx_f, approx_g)
    x_max = x_out[~np.isnan(y_max)]
    y_max = y_max[~np.isnan(y_max)]

    # compute AUVs and PSO
    AUV_g = np.trapz(y_out_g, x_out_g)  # y,x
    AUV_f = np.trapz(y_out_f, x_out_f)
    AUV_min = np.trapz(y_min, x_min)
    AUV_max = np.trapz(y_max, x_max)
    # PSO = AUV_min / (AUV_g + AUV_f)
    PSO = AUV_min / AUV_max
    # return PSO
    pso.append({'file': f, 'PSO': PSO, 'AUVg': AUV_g, 'AUVf': AUV_f, 'AUVmin': AUV_min})

pso = pd.DataFrame(pso)

# %%
pso

# %% [markdown]
# Load extra data such as pathdev from `'data/moves_and_deviation.csv'` for modeling and plotting

# %%
df_path = pd.read_csv('/Users/neako/Documents/Cours-MasCo/PhD/tools/githubs/IT/acl2017-spectral-master/data/moves_and_deviation.csv')

psom = pd.merge(left = pso, right = df_path, left_on="file", right_on="Observation")
psom.head()

# %% [markdown]
# How to add intercepts: https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

# %%
import statsmodels.api as sm

# %%
model = sm.OLS(psom['path dev'], psom['PSO']).fit() ## sm.OLS(output, input)
# Print out the statistics
model.summary()

# %%
sns.jointplot(x="PSO", y="path dev", data=psom, kind="reg")

# %%


# %% [markdown]
# **RP**
# ```R
# dt.peakPS = dt[, {
#         y_a = ent_swbd[who=='f']
#         y_b = ent_swbd[who=='g']
#         len = min(length(y_a), length(y_b))
#         y_a = y_a[1:len]
#         y_b = y_b[1:len]
#         comb.ts = ts(matrix(c(y_a, y_b), ncol=2))
#         spec = spectrum(comb.ts, detrend=FALSE, taper=0, log='no', plot=F)
#         # phase shift at all peaks
#         i_max_a = which(diff(sign(diff(spec$spec[,1])))<0) + 1
#         i_max_b = which(diff(sign(diff(spec$spec[,2])))<0) + 1
#         peakPS = spec$phase[,1][union(i_max_a, i_max_b)]
#         # return
#         .(peakPS = peakPS)
#     }, by = observation]
# dt.peakPS = dt.peakPS[dt.dev[, .(observation, pathdev)], nomatch=0]
# ```

# %%
col = 'xu_h'

# %%
rp = []
for f in df.file.unique():
    max_len = min(df[df.file == f].groupby('who').count()['xu_h'].tolist())
    y_f = df[(df.file == f) & (df.who == 'f')][col].iloc[:max_len]
    y_g = df[(df.file == f) & (df.who == 'g')][col].iloc[:max_len]
    comb_ts = pd.concat([y_f.reset_index(drop=True), y_g.reset_index(drop=True)], axis=1) #, ignore_index=True - not working
    spec = spec_pgram(comb_ts, taper=0, plot=False, log='no', detrend=False)
    spec_df = pd.DataFrame(spec['spec'])

    # phase shift at all peaks
    i_max = np.sign(spec_df.diff()).diff().apply(lambda x: np.where(x < 0)[0], axis=0)
    i_index = np.array(list(set(np.concatenate((i_max[0],i_max[1]))))) - 1
    peakPS = spec['phase'].T[0][i_index]
    rp.append({'file': f, 'peakPS': peakPS})

rp = pd.DataFrame(rp).explode('peakPS')


# %%
df_path = pd.read_csv('/Users/neako/Documents/Cours-MasCo/PhD/tools/githubs/IT/acl2017-spectral-master/data/moves_and_deviation.csv')
df_path_match = df_path.set_index('Observation')['path dev'].to_dict()

rp['pathdev'] = rp.file.apply(lambda x: np.nan if x not in df_path_match else df_path_match[x])
rp[~np.isnan(rp.pathdev)] # OK shape

# %%
# Compute mean, median and max values for peakPS
rp_mean = rp.groupby('file').agg({
    'pathdev':'mean', 
    'peakPS': [lambda x: np.mean(np.abs(x)), lambda x: np.median(np.abs(x)), lambda x: np.max(np.abs(x))]
}).droplevel(0, axis=1)
rp_mean.columns = ['pathdev', 'mean', 'median', 'max']
rp_mean

# %% [markdown]
# TODO: models

# %%



