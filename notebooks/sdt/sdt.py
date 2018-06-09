import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style='ticks', font_scale=1.5)
blue, green, red, purple = sns.color_palette("deep",4)
gray, black = (.25,.25,.25), (0,0,0)

fig_dict = {'colors':{'N':blue, 'S':red, 'lam':gray}, 'response_labs':{'N':'"No"','S':'"Yes"'},
            'xl':4, 'xtick_labs':[r'$\mu_N$',r'$\lambda$',r'$\mu_S$'], 'figsize':(17,5),
            'shade':['h','f'], 'legend':['Noise','Signal + Noise','Response Criterion']}

def plot_simple_model(pdf=norm.pdf, mu=[0,1], sd=[1,1], lam=.5, fig_dict=fig_dict, ax=None):
  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=fig_dict['figsize'])
  mu_N = mu[0]
  mu_S = mu[1]
  sd_N = sd[0]
  sd_S = sd[1]
  xl = fig_dict['xl']
  pe = np.linspace(mu_N-xl*sd_N, mu_S+xl*sd_S, 500) # perceptual evidence scale
  fn = pdf(pe, loc=mu_N, scale=sd_N) # noise-only perceptual distribution
  fs = pdf(pe, loc=mu_S, scale=sd_S) # signal + noise perceptual distribution
  lam_i = np.argmin(np.abs(pe-lam)) # response criterion & index
  clrs = fig_dict['colors']
  ls, = ax.plot(pe, fs, color=clrs['S'], lw=3)
  if 'h' in fig_dict['shade']:
    ax.fill_between(x=pe[lam_i:], y1=np.zeros(len(pe))[lam_i:], y2=fs[lam_i:], color=clrs['S'], alpha=.5)
  ln, = ax.plot(pe, fn, color=clrs['N'], lw=3)
  if 'f' in fig_dict['shade']:
    ax.fill_between(x=pe[lam_i:], y1=np.zeros(len(pe))[lam_i:], y2=fn[lam_i:], color=blue, alpha=.5)
  ll, = ax.plot([lam,lam], [0,np.max((fs[lam_i],fn[lam_i]))], '-', lw=3, color=clrs['lam'])
  rl = fig_dict['response_labs']
  rl_y = np.max((fn.max(), fs.max()))*2/3
  ax.text(mu_N-1.5, rl_y, rl['N'], ha='center', fontsize=18)
  ax.text(mu_S+1.5, rl_y, rl['S'], ha='center', fontsize=18)
  leg = fig_dict['legend']
  if leg:
    ax.legend([ln,ls,ll],leg)
  xtt = np.array([mu_N, lam, mu_S])
  xtl = np.array(fig_dict['xtick_labs'])
  sort_idx = np.argsort(xtt)
  ax.set(xticks=xtt[sort_idx], yticks=[])
  ax.set_xticklabels(xtl[sort_idx], fontsize=18)
  sns.despine(ax=ax, top=True, left=True, right=True)
