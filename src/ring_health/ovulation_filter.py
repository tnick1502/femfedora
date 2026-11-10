"""Causal Bayesian timing baseline with explicit assumed likelihood.

Provides a coherent distribution over ovulation days, including no ovulation.
It is a mechanistic baseline; it has not been fitted to prospective Oura data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, t


def ovulation_posterior(daily: pd.DataFrame, period_starts: list[str], as_of=None) -> dict:
    daily=daily.copy()
    daily["date"]=pd.to_datetime(daily.date)
    today=pd.Timestamp(as_of) if as_of is not None else daily.date.max()
    starts=pd.DatetimeIndex(pd.to_datetime(period_starts)).sort_values().unique()
    if (starts>today).any():
        raise ValueError("Future period starts are not permitted")
    if len(starts)<4:
        raise ValueError("Need at least3 completed historical cycle lengths")
    lengths=np.diff(starts.values).astype("timedelta64[D]").astype(int)[-6:]
    if (lengths<19).any() or (lengths>65).any():
        raise ValueError("Cycle history outside demo support19..65days")
    start=starts[-1]
    day=int((today-start).days)+1
    if not 7<=day<=60:
        raise ValueError("Current cycle day outside timing-filter support7..60")
    c=daily[(daily.date>=start)&(daily.date<=today)].copy()
    c["cycle_day"]=(c.date-start).dt.days+1
    baseline=c[c.cycle_day<=6].temperature_deviation.dropna()
    if len(baseline)<4:
        raise ValueError("Need4 valid temperature nights in first6days of current cycle")
    observed=c[(c.cycle_day>=7)&c.temperature_deviation.notna()]
    if len(observed)<3:
        raise ValueError("Need3 observed temperatures after baseline")
    grid=np.arange(6,61)
    prior=norm.pdf(grid,loc=np.median(lengths)-13,scale=max(3.,float(np.std(lengths))))
    prior=.90*prior/prior.sum()
    prior=np.r_[prior,.10]
    # Correlated nights: temper likelihood to limit overconfidence. Likelihood
    # scale and0.4C rise are engineering assumptions within literature ranges.
    days=observed.cycle_day.to_numpy()
    y=observed.temperature_deviation.to_numpy()-float(baseline.median())
    mu=.4*np.clip((days[:,None]-grid[None,:])/2,0,1)
    mu=np.column_stack([mu,np.zeros(len(days))])
    loglik=t.logpdf((y[:,None]-mu)/.20,df=4).sum(axis=0)*.30
    logpost=np.log(prior+1e-300)+loglik
    posterior=np.exp(logpost-logsumexp(logpost))
    conditional=posterior[:-1]/posterior[:-1].sum()
    q=[int(grid[min(np.searchsorted(np.cumsum(conditional),a),len(grid)-1)]) for a in [.05,.5,.95]]
    forecast=[]
    for horizon in range(7):
        future_day=day+horizon
        mask=(grid>=future_day)&(grid<=future_day+5)
        forecast.append({"date":str((today+pd.Timedelta(days=horizon)).date()),"cycle_day":future_day,
                         "fertile_day_probability":float(posterior[:-1][mask].sum())})
    return {"model":"bayesian_timing_temperature_baseline_v1","provenance":"mechanistic_assumptions_unvalidated",
            "ovulation_cycle_day_conditional_quantiles_05_50_95":q,
            "no_ovulation_component_probability":float(posterior[-1]),
            "ovulation_already_occurred_probability":float(posterior[:-1][grid<day].sum()),
            "forecast":forecast,
            "interpretation":"Probability of belonging to the six-day current-cycle window, not probability of pregnancy. Future window before temperature rise is mainly driven by historical cycle timing."}
