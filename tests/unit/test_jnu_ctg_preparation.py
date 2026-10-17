import numpy as np
import pytest
from src.features.jnu_ctg_preparation import basic_stats,bounded_interpolation,signal_masks,signal_features


def test_statistics_units_and_quantiles():
    x=np.array([1.,2.,3.,4.,5.]);s=basic_stats(x,'x_')
    assert len(s)==11
    assert s['x_q95']==pytest.approx(np.quantile(x,.95))
    assert s['x_iqr']==pytest.approx(2.) and s['x_mad']==pytest.approx(1.)


def test_bounded_interpolation_never_bridges_long_gaps_or_edges():
    x=np.array([0.,120.,0.,0.,126.,0.,0.,0.,0.,130.,0.])
    y,filled=bounded_interpolation(x,x>0,max_gap_samples=2)
    np.testing.assert_allclose(y[2:4],[122.,124.])
    assert np.isnan(y[0]) and np.isnan(y[-1]) and np.isnan(y[5:9]).all()
    assert np.flatnonzero(filled).tolist()==[2,3]


def test_missing_masks_keep_true_uc_zero_and_extreme_fhr():
    wave=np.column_stack([np.full(240,130.),np.zeros(240)])
    wave[0,0]=0;wave[2,0]=260;wave[4,0]=-1
    masks,_=signal_masks(wave)
    assert not masks['fhr_observed'][0] and not masks['fhr_observed'][4]
    assert masks['fhr_source_present'][2] and masks['fhr_range_flag'][2]
    assert not masks['fhr_observed'][2] and wave[2,0]==260
    assert masks['uc_observed'].all() and masks['uc_zero'].all()


def test_no_adjacent_difference_across_missing_fhr():
    wave=np.column_stack([np.full(240,130.),np.zeros(240)])
    wave[:120,0]=120;wave[120,0]=0;wave[121:,0]=160
    features,quality,_,_=signal_features(wave)
    assert features['fhr_adjacent_mean_abs_bpm']==0
    assert quality['fhr_gap_count']==1
    assert 'STV' not in features


def test_all_missing_channel_remains_missing_in_descriptors():
    wave=np.zeros((240,2));features,quality,_,_=signal_features(wave)
    assert np.isnan(features['fhr_bpm_mean'])
    assert np.isnan(features['project_rr_epoch_adjacent_abs_ms'])
    assert features['uc_device_mean']==0
    assert quality['fhr_observed_fraction']==0


def test_constant_runs_flagged_without_erasing_potential_low_variability():
    wave=np.column_stack([np.full(240,130.),np.zeros(240)])
    features,quality,masks,_=signal_features(wave)
    assert quality['fhr_longest_constant_valid_run_s']==60
    assert quality['fhr_fraction_in_constant_runs_30s']==1
    assert features['fhr_bpm_mean']==130 and masks['fhr_observed'].all()


def test_preprocessor_rejects_labels_and_handles_unseen_missingness(tmp_path):
    import pandas as pd,joblib
    from src.features.jnu_ctg_preparation import preprocessor_template
    x=pd.DataFrame({'a':[1.,2.,3.],'b':[10.,20.,30.]})
    pipeline=preprocessor_template(['a','b']).fit(x)
    transformed=pipeline.transform(pd.DataFrame({'a':[np.nan],'b':[20.]}))
    assert transformed.shape==(1,4) and transformed[0,2:].tolist()==[1.,0.]
    scale=pipeline.named_steps['columns'].named_transformers_['scaled'].named_steps['scaler']
    np.testing.assert_allclose(scale.center_,[2.,20.]);np.testing.assert_allclose(scale.scale_,[1.,10.])
    center=scale.center_.copy();pipeline.transform(pd.DataFrame({'a':[1e6],'b':[1e8]}))
    np.testing.assert_array_equal(center,scale.center_)
    with pytest.raises(ValueError,match='labels'):pipeline.transform(x.assign(figo_label=0))
    p=tmp_path/'prep.joblib';joblib.dump(pipeline,p)
    np.testing.assert_array_equal(joblib.load(p).transform(x),pipeline.transform(x))


def test_signal_normalizer_fits_selected_rows_and_returns_masks():
    from src.features.jnu_ctg_preparation import JNUSignalNormalizer
    x=np.empty((3,7200,2));x[0,:,0]=120;x[1,:,0]=140;x[2,:,0]=200
    x[:2,:,1]=np.tile([0.,20.],3600);x[2,:,1]=100
    norm=JNUSignalNormalizer().fit_rows(x,[0,1])
    np.testing.assert_allclose(norm.mean_,[130.,10.]);np.testing.assert_allclose(norm.scale_,[10.,10.])
    hold=x[2].copy();hold[0,0]=0;hold[10:14,0]=0;hold[50:100,0]=0;hold[200,0]=255
    out=norm.transform_one(hold)
    assert not out['used_mask'][0,0] and out['values'][0,0]==0
    assert out['interpolated_mask'][10:14,0].all()
    assert not out['used_mask'][50:100,0].any()
    np.testing.assert_allclose(norm.mean_,[130.,10.])
    assert out['observed_mask'][:,1].all()


def test_patient_and_duplicate_signal_boundaries(monkeypatch,tmp_path):
    import pandas as pd
    import src.features.jnu_ctg_preparation as prep
    monkeypatch.setattr(prep,'OUT',tmp_path);monkeypatch.setattr(prep,'PROCESSED',tmp_path)
    records=[];index=[]
    for patient in range(60):
        for j in range(2):
            rid=f'{patient}_{j}'
            records.append(dict(record_id=rid,patient_id=patient,figo_label=j,neonatal_asphyxia=1 if patient%7==0 else 0))
            index.append(dict(record_id=rid,fhr_sha256='shared' if patient in [0,1] and j==0 else rid,waveform_sha256=rid))
    split,_=prep.split_patients(pd.DataFrame(records),pd.DataFrame(index))
    assert split.groupby('patient_id').split.nunique().max()==1
    assert split.loc[split.patient_id.isin([0,1]),'split'].nunique()==1
    assert set(split.split)=={'train','validation','test'}
    assert split.groupby('patient_id').inverse_patient_record_count.sum().eq(1).all()
