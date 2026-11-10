import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ring_health.features import SIGNALS, daily_features, oura_to_daily
from src.ring_health.ovulation_filter import ovulation_posterior
from src.ring_health.predict import predict
from src.ring_health.simulation import fertility_scenarios


class RingHealthTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"date":pd.date_range("2024-01-01",periods=100),"temperature_deviation":np.sin(np.arange(100)/5)})

    def test_causal_future_mutation(self):
        d=self.frame()
        a=daily_features(d).iloc[:50]
        d.loc[50:,"temperature_deviation"]=1000
        pd.testing.assert_frame_equal(a,daily_features(d).iloc[:50])

    def test_no_compression_of_missing_days(self):
        d=self.frame().drop(index=[10,11,12])
        f=daily_features(d)
        self.assertEqual(len(f),100)
        self.assertTrue(f.loc[11,"temperature_deviation_observed7"]<1)

    def test_duplicate_days_rejected(self):
        with self.assertRaises(ValueError):daily_features(pd.concat([self.frame(),self.frame().iloc[:1]]))

    def test_baseline_is_previous_nights(self):
        d=pd.DataFrame({"temperature_deviation":list(range(10))})
        f=daily_features(d)
        # On day index5, prior values0..4 median2; current5 minus2=3.
        self.assertEqual(f.loc[5,"temperature_deviation_delta"],3)

    def test_oura_main_sleep_and_units(self):
        p={"daily_readiness":[{"day":"2024-01-02","temperature_deviation":.2}],"daily_activity":[{"day":"2024-01-02","steps":4000}],
           "sleep":[{"day":"2024-01-02","type":"long_sleep","total_sleep_duration":24000,"lowest_heart_rate":55,"average_hrv":40,"average_breath":14,"efficiency":90},
                    {"day":"2024-01-02","type":"rest","total_sleep_duration":1000},
                    {"day":"2024-01-02","type":"long_sleep","total_sleep_duration":18000}]}
        f=oura_to_daily(p)
        self.assertEqual(len(f),1)
        self.assertEqual(f.sleep_minutes.iloc[0],400)
        self.assertEqual(f.temperature_deviation.iloc[0],.2)
        self.assertEqual(f.resting_hr.iloc[0],55)

    def test_unknown_context_abstains(self):
        self.assertEqual(predict({})["status"],"ineligible_or_unknown_context")

    def test_no_recommendations(self):
        self.assertFalse(predict({})["medical_recommendation"])

    def test_future_observations_rejected(self):
        flags={k:False for k in ["pregnant","breastfeeding","hormonal_medication","hysterectomy","bilateral_oophorectomy"]}
        with self.assertRaises(ValueError):
            predict({"as_of":"2024-01-01","context":{"age":40,**flags},"daily":[{"date":"2024-01-02","temperature_deviation":.2}]})

    def test_bad_units_rejected(self):
        flags={k:False for k in ["pregnant","breastfeeding","hormonal_medication","hysterectomy","bilateral_oophorectomy"]}
        out=predict({"context":{"age":40,**flags},"daily":[{"date":"2024-01-02","temperature_deviation":36.5}]})
        self.assertEqual(out["status"],"out_of_range_values")

    def test_bayesian_probability_coherence(self):
        d=self.frame()
        starts=["2024-01-01","2024-01-29","2024-02-26","2024-03-25"]
        r=ovulation_posterior(d,starts)
        self.assertTrue(all(0<=x["fertile_day_probability"]<=1 for x in r["forecast"]))
        self.assertEqual(len(r["forecast"]),7)
        q=r["ovulation_cycle_day_conditional_quantiles_05_50_95"]
        self.assertEqual(q,sorted(q))

    def test_bayesian_ignores_unseen_future(self):
        d=self.frame()
        starts=["2024-01-01","2024-01-29","2024-02-26","2024-03-25"]
        a=ovulation_posterior(d,starts,as_of="2024-04-06")
        d.loc[d.date>"2024-04-06","temperature_deviation"]=1000
        self.assertEqual(a,ovulation_posterior(d,starts,as_of="2024-04-06"))

    def test_future_period_rejected(self):
        with self.assertRaises(ValueError):ovulation_posterior(self.frame(),["2024-01-01","2024-01-29","2024-02-26","2025-01-01"])

    def test_simulation_seed_repeatable(self):
        a,_,_=fertility_scenarios(n_people=2,seed=12)
        b,_,_=fertility_scenarios(n_people=2,seed=12)
        pd.testing.assert_frame_equal(a,b)

    def test_demo_example_end_to_end(self):
        path=Path(__file__).resolve().parents[2]/"data/samples/ring_health/fertility_demo_input.json"
        result=predict(json.loads(path.read_text()))
        self.assertEqual(result["fertility_demo"]["status"],"scored")
        self.assertEqual(result["fertility_demo"]["provenance"],"synthetic_only")
        self.assertTrue(0<=result["fertility_demo"]["fertile_day_probability"]<=1)


if __name__=="__main__":unittest.main()
