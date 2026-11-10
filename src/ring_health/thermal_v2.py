"""Gap-aware event-time likelihood, with missing observations marginalized.

Offsets and slow drift are Gaussian nuisance parameters, integrated out.
No value is written into an unobserved night. All conditioning is as-of.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


def posterior(days, temperatures, grid, prior, profile, *, prior_only=False):
    days=np.asarray(days,dtype=float)
    y=np.asarray(temperatures,dtype=float)
    grid=np.asarray(grid,dtype=float)
    prior=np.asarray(prior,dtype=float)
    valid=np.isfinite(y)&np.isfinite(days)
    days,y=days[valid],y[valid]
    if len(y)<4 or prior_only:
        return prior/prior.sum()
    amplitudes=np.asarray(profile["amplitudes"],dtype=float)
    amp_weights=np.asarray(profile["amplitude_weights"],dtype=float)
    # Robust clipping about the observed median bounds fever/outlier influence.
    center=float(np.median(y))
    y=np.clip(y-center,-.7,.7)
    sigma=profile["noise_sigma"]
    rho=profile.get("rho",.5)
    distance=np.abs(days[:,None]-days[None,:])
    covariance=sigma**2*rho**distance+np.eye(len(days))*(sigma*.55)**2
    # Integrate offset and gradual non-cycle drift; a missing baseline is OK.
    time=(days-days.mean())/28
    covariance+=.5**2*np.ones_like(covariance)+.12**2*np.outer(time,time)
    shape=np.clip((days[:,None]-grid[None,:])/profile.get("rise_days",2),0,1)
    expected=(shape[:,:,None]*amplitudes[None,None,:]).reshape(len(days),-1)
    residual=y[:,None]-expected
    inv=np.linalg.inv(covariance)
    ll=-.5*np.sum(residual*(inv@residual),axis=0).reshape(len(grid),len(amplitudes))
    event_ll=logsumexp(ll+np.log(amp_weights)[None,:],axis=1)
    null_ll=-.5*float(y@inv@y)
    # Correlation is explicit, plus a conservative tempering parameter.
    likelihood=np.r_[event_ll,null_ll]*profile.get("likelihood_weight",.5)
    logp=np.log(np.maximum(prior,1e-300))+likelihood
    return np.exp(logp-logsumexp(logp))


def timing_prior(grid, previous_lengths, current_day, *, logs_complete=False):
    lengths=np.asarray(previous_lengths,dtype=float)
    lengths=lengths[(lengths>=19)&(lengths<=50)]
    n=len(lengths)
    # Shrink sparse history toward published population moments; history is not
    # mandatory and population variability must not shrink to zero.
    weight=n/(n+3)
    median=float(np.median(lengths)) if n else 29.3
    mean=weight*(median-12.4)+(1-weight)*16.9
    sd=max(3.,float(np.std(lengths)) if n>=3 else 5.3)
    p=norm.pdf(grid,mean,sd)
    if logs_complete:
        # Only explicitly complete logs permit interpreting no new entry as
        # no next menses. Joint luteal-duration survival downweights old events.
        p*=norm.sf(current_day-np.asarray(grid),loc=12.4,scale=2.4)
    p=p/max(p.sum(),1e-300)*.90
    return np.r_[p,.10]


def analyze_cycle(frame, period_starts, as_of, profile, *, logs_complete=False):
    import pandas as pd
    today=pd.Timestamp(as_of)
    starts=pd.DatetimeIndex(pd.to_datetime(period_starts)).sort_values().unique()
    if not len(starts):
        return {"status":"anchor_required","required":"One known period start or current cycle_day; historical cycles and LH tests are optional"}
    start=starts[-1]
    day=int((today-start).days)+1
    if day>50:
        return {"status":"cycle_out_of_support" if logs_complete else "uncertain_cycle_anchor","reason":"More than50days since last start; model support exceeded or an unlogged period is possible","cycle_day":day}
    if not 1<=day<=50:
        return {"status":"cycle_out_of_support","cycle_day":day}
    lengths=np.diff(starts.values).astype("timedelta64[D]").astype(int)[-6:]
    # Long intervals may include missing logs; do not silently divide or impute.
    uncertain_intervals=int(np.sum((lengths<19)|(lengths>50)))
    c=frame[(frame.date>=start)&(frame.date<=today)].copy()
    c["cycle_day"]=(c.date-start).dt.days+1
    observed=c[c.temperature_deviation.notna()]
    days=observed.cycle_day.to_numpy()
    y=observed.temperature_deviation.to_numpy()
    latest_gap=day-int(days.max()) if len(days) else day
    # Support includes the entire forecast horizon and its possible ovulation.
    grid=np.arange(5,max(80,day+13)+1)
    prior=timing_prior(grid,lengths,day,logs_complete=logs_complete)
    evidence_usable=len(y)>=8 and (days.max()-days.min()>=10 if len(days) else False)
    p=posterior(days,y,grid,prior,profile,prior_only=not evidence_usable or latest_gap>10)
    conditional=p[:-1]/max(p[:-1].sum(),1e-300)
    q=[int(grid[min(np.searchsorted(np.cumsum(conditional),x),len(grid)-1)]) for x in [.05,.5,.95]]
    estimated=int(grid[np.argmax(p[:-1])])
    before=int(np.sum(days<=estimated-1))
    after=int(np.sum(days>=estimated+2))
    pre=y[(days>=estimated-8)&(days<=estimated-1)]
    post=y[(days>=estimated+2)&(days<=min(day,estimated+8))]
    shift=float(np.median(post)-np.median(pre)) if len(pre)>=3 and len(post)>=3 else None
    past_probability=float(p[:-1][grid<day].sum())
    # A previous thermal event does not establish the current cycle anchor.
    # In incomplete logs, a new unlogged menses can otherwise make an old
    # event look like a current-cycle detection. Fourteen days is a conservative
    # operational horizon, not a learned or clinically validated luteal cutoff.
    if not logs_complete and evidence_usable and past_probability>=.90 and day-estimated>14:
        return {"status":"uncertain_cycle_anchor","cycle_day":day,
                "reason":"Thermal event is historical and period logs are incomplete; update the current period start or cycle_day",
                "reference_event_date":None,"imputed_measurement_count":0,
                "missing_observations_method":"marginalized_in_likelihood"}
    detected=bool(evidence_usable and latest_gap<=4 and past_probability>=.90 and q[2]-q[0]<=8 and before>=3 and after>=3 and shift is not None and shift>=profile.get("detection_min_shift",.1))
    status="thermal_shift_detected" if detected else ("estimated_with_gaps" if evidence_usable else "calendar_only")
    if latest_gap>10:status="calendar_only_stale_measurements"
    forecast=[{"date":str((today+pd.Timedelta(days=h)).date()),"fertile_day_probability":float(p[:-1][(grid>=day+h)&(grid<=day+h+5)].sum())} for h in range(7)]
    return {"status":status,"cycle_day":day,"evidence_source":"observed_temperature_and_cycle_prior" if evidence_usable and latest_gap<=10 else "cycle_prior_only",
            "reference_event_date":str((start+pd.Timedelta(days=estimated-1)).date()) if detected else None,
            "event_day_conditional_quantiles_05_50_95":q,"event_day_quantiles_are_clinically_calibrated":False,
            "postovulatory_component_probability":past_probability,"no_ovulation_component_probability":float(p[-1]),
            "observed_temperature_nights":len(y),"days_since_last_temperature":latest_gap,"uncertain_history_intervals":uncertain_intervals,
            "missing_observations_method":"marginalized_in_likelihood","imputed_measurement_count":0,
            "night_counts_before_after_estimate":[before,after],"observed_shift_celsius":shift,"forecast":forecast,
            "interpretation":"Model hypothesis of a sustained thermal shift; neither hormone-confirmed ovulation nor a pregnancy probability"}
