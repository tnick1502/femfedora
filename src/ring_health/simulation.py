"""Mechanistic scenario generator, NOT a fitted population model.

Literature supports directions and biphasic temperature changes. Distributions,
stage effects, confounding strengths and error mechanisms below are explicit
engineering assumptions. Simulator accuracy is never clinical accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import SIGNALS, daily_features, window_features


def ar_noise(rng, n, sigma, rho=.65):
    e = rng.normal(0, sigma, n)
    for k in range(1, n):
        e[k] += rho * e[k-1]
    return e


def simulate_signals(rng, n, luteal, age, burden=0., stress=0., shift=0., noise=1., missing=.07):
    individual_temp_amplitude = rng.uniform(.15, .35)
    # Uniform step from0.3 to0.7C peak-to-peak, then individual attenuation.
    physiology = luteal - np.mean(luteal[:min(28, n)])
    illness = np.zeros(n)
    if rng.random() < .35:
        start = int(rng.integers(8, max(9, n-7)))
        illness[start:start+int(rng.integers(2, 6))] = rng.uniform(.25, .9)
    shared = ar_noise(rng, n, .6 * noise)
    temp = individual_temp_amplitude * 2 * physiology + ar_noise(rng, n, .07*noise) + illness + shift
    hr = rng.normal(61, 7) + .08*(age-40) + rng.uniform(1.5, 4)*luteal + burden + 2*shared + illness*8
    hrv = np.exp(rng.normal(3.7, .30) - .07*luteal - .02*burden - .06*shared + ar_noise(rng,n,.06*noise))
    rr = rng.normal(15, 1.2) + .25*luteal + .10*shared + illness*.7
    sleep = rng.normal(435, 28) - burden*3 - stress*8 + rng.normal(0, 27*noise, n) - shared*5
    efficiency = np.clip(rng.normal(89, 3) - burden*.4 - stress + rng.normal(0, 3*noise, n), 40, 100)
    steps = np.maximum(300, rng.normal(6500, 1800) + rng.normal(0, 1800, n) - stress*300 - illness*1000)
    d = pd.DataFrame(dict(zip(SIGNALS,[temp,hr,hrv,rr,sleep,efficiency,steps])))
    d.insert(0,"date", pd.date_range("2024-01-01", periods=n))
    nonwear = rng.random(n) < missing
    d.loc[nonwear, SIGNALS] = np.nan
    if rng.random() < .2:
        k = int(rng.integers(5, max(6,n-10)))
        d.loc[k:k+int(rng.integers(2,6)), SIGNALS] = np.nan
    return d


def stage_scenarios(n_people=1500, seed=704, stress_test=False):
    rng = np.random.default_rng(seed)
    rows, samples = [], []
    for person in range(n_people):
        stage = int(rng.choice(3, p=[.36,.32,.32]))
        age = float(np.clip(rng.normal([43,49,55][stage], [6,5,5][stage]),35,65))
        n = 120
        luteal = np.zeros(n)
        pos = -int(rng.integers(0,30))
        while pos < n:
            length = int(np.clip(rng.normal([28,35,35][stage],[3,12,12][stage]),19,75))
            ovulatory = rng.random() < [.92,.60,.00][stage]
            if ovulatory:
                ov = pos + length - int(rng.integers(11,16))
                luteal[max(0,ov+1):max(0,min(n,pos+length))] = 1
            pos += length
        # Burden overlaps heavily; neither all peri nor all post have symptoms.
        burden = max(0.,rng.normal([.4,1.6,1.2][stage],1.0))
        d = simulate_signals(rng,n,luteal,age,burden=burden, stress=rng.uniform(0,3),
                             noise=2.0 if stress_test else 1., shift=.15 if stress_test else 0.,
                             missing=.30 if stress_test else .07)
        if stress_test:
            d["temperature_deviation"] *= .55
        row = window_features(d, age)
        row.update(participant_id=f"stage_{seed}_{person}",target=stage,provenance="synthetic")
        rows.append(row)
        if person < 6:
            d["participant_id"] = row["participant_id"]
            d["target_stage"] = stage
            samples.append(d)
    return pd.DataFrame(rows), pd.concat(samples,ignore_index=True)


def fertility_scenarios(n_people=360, seed=905, stress_test=False, include_all_daily=False):
    rng = np.random.default_rng(seed)
    allrows, events, samples = [], [], []
    for person in range(n_people):
        age = rng.uniform(22,43)
        typical = float(np.clip(rng.normal(29,3.5),22,40))
        variability = rng.choice([2.,5.,9.],p=[.60,.27,.13])
        # Only previous completed cycles inform the current-cycle prior.
        history = list(np.clip(rng.normal(typical,variability,6).round().astype(int),19,65))
        pos, segments, luteal, all_days = 0, [], [], []
        for cycle in range(5):
            length = int(np.clip(rng.normal(typical,variability),19,65))
            ovulatory = rng.random() > (.12 if stress_test else .07)
            ov = length - int(np.clip(rng.normal(13,2),9,18)) if ovulatory else -100
            ov = max(6,ov) if ovulatory else ov
            segments.append((cycle,pos,length,ov,ovulatory,list(history)))
            luteal.extend([int(ovulatory and day>ov) for day in range(1,length+1)])
            all_days.extend(range(1,length+1))
            history = history[1:] + [length]
            pos += length
        d = simulate_signals(rng,pos,np.asarray(luteal),age,stress=rng.uniform(0,3),noise=1.8 if stress_test else 1.,missing=.25 if stress_test else .07)
        if stress_test:
            d.temperature_deviation *= .6
        feat = daily_features(d)
        for cycle,start,length,ov,ovulatory,hist in segments:
            cid = f"fert_{seed}_{person}_{cycle}"
            # First cycle is warmup only, no labels or per-cycle baselines in X.
            if cycle == 0:
                continue
            g = feat.iloc[start:start+length].copy()
            day = np.arange(1,length+1)
            g["cycle_day"] = day
            g["previous_cycle_median"] = float(np.median(hist))
            g["previous_cycle_std"] = float(np.std(hist))
            g["age"] = age
            g["fertile_target"] = (ovulatory & (day >= ov-5) & (day <= ov)).astype(int)
            g["postovulatory_target"] = (ovulatory & (day >= ov+2)).astype(int)
            g["ovulatory"] = ovulatory
            g["ovulation_day"] = ov if ovulatory else np.nan
            g["participant_id"] = f"fert_{seed}_{person}"
            g["cycle_id"] = cid
            g["provenance"] = "synthetic"
            allrows.append(g)
            events.append({"participant_id":g.participant_id.iloc[0],"cycle_id":cid,"ovulation_day":ov if ovulatory else None,"length":length,"ovulatory":ovulatory})
        if person < 4 or include_all_daily:
            d["participant_id"] = f"fert_{seed}_{person}"
            d["cycle_day"] = all_days
            d["period_start"] = np.asarray(all_days)==1
            d["ovulation_day"] = np.concatenate([np.full(length,ov if ovulatory else np.nan) for _,_,length,ov,ovulatory,_ in segments])
            d["cycle_id"] = np.concatenate([np.full(length,f"fert_{seed}_{person}_{cycle}") for cycle,_,length,_,_,_ in segments])
            samples.append(d)
    return pd.concat(allrows,ignore_index=True),pd.DataFrame(events),pd.concat(samples,ignore_index=True)
