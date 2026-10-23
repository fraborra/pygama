"""Regression tests for optional Bayesian energy-calibration backends."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from pygama.math.distributions import gauss_on_step
from pygama.pargen import energy_cal


@pytest.fixture
def calibration():
    cal = energy_cal.HPGeCalibration("energy", [100.0, 200.0], 0.5, deg=0)
    cal.peak_locs = np.array([200.0, 400.0])
    return cal


def peak_draws(mode_guess=200.0):
    offsets = np.linspace(-0.3, 0.3, 8)
    values = {
        "x_lo": mode_guess - 20,
        "x_hi": mode_guess + 20,
        "n_sig": 1000,
        "mu": mode_guess + 1,
        "sigma": 2,
        "n_bkg": 100,
        "hstep": 0.1,
    }
    draws = {
        name: np.full(8, values[name], dtype=float)
        for name in gauss_on_step.required_args()
    }
    draws["mu"] += offsets
    draws["mode"] = mode_guess + offsets
    return draws


def curve_result(size=2):
    scale = np.zeros(size)
    scale[1] = 2.0
    calibration = np.zeros(size)
    calibration[1] = 0.5
    uncertainties = np.zeros(size)
    uncertainties[1] = 0.01
    covariance = np.diag(uncertainties**2)
    return {
        "pk_cal_pars": scale,
        "pk_cal_errs": uncertainties,
        "pk_cal_cov": covariance,
        "calibration_parameters": calibration,
        "calibration_uncertainties": uncertainties,
        "calibration_covariance": covariance,
    }


def fit_peaks(cal, **kwargs):
    # Include out-of-window events to check selection and truncation.
    energies = np.array([170, 195, 198, 200, 202, 230, 370, 395, 398, 400, 402, 430])
    cal.hpge_fit_energy_peaks(
        energies,
        peak_pars=[(peak, (10, 10), gauss_on_step) for peak in cal.peaks_kev],
        **kwargs,
    )


def test_fitters_must_be_configured_together(calibration):
    for fitters in [(Mock(), None), (None, Mock())]:
        with pytest.raises(ValueError, match="configured together"):
            calibration.set_bayesian_fitters(*fitters)


@pytest.mark.parametrize("peak_param", ["mode", "mu"])
def test_bayesian_peak_and_curve_integration(calibration, monkeypatch, peak_param):
    artifact = object()
    peak_fitter = Mock(
        side_effect=lambda **kwargs: (peak_draws(kwargs["mode_guess"]), artifact)
    )
    expected_curve = curve_result()
    curve_fitter = Mock(return_value=(expected_curve, artifact))
    calibration.set_bayesian_fitters(peak_fitter, curve_fitter)
    frequentist = Mock(side_effect=AssertionError("unexpected frequentist fit"))
    monkeypatch.setattr(energy_cal, "unbinned_staged_energy_fit", frequentist)
    monkeypatch.setattr(energy_cal, "hpge_fit_energy_scale", frequentist)
    monkeypatch.setattr(energy_cal, "hpge_fit_energy_cal_func", frequentist)

    fit_peaks(calibration, peak_param=peak_param, n_events=3)

    result = calibration.results["hpge_fit_energy_peaks"]
    np.testing.assert_array_equal(
        calibration.pars, expected_curve["calibration_parameters"]
    )
    for key, value in expected_curve.items():
        np.testing.assert_array_equal(result[key], value)
    assert result["peak_param"] == peak_param
    assert calibration.bayesian_calibration_posteriors == [artifact]
    assert len(calibration.bayesian_peak_posteriors) == 1
    assert len(calibration.bayesian_peak_fits) == 1
    assert curve_fitter.call_args.kwargs["degree"] == 0
    assert curve_fitter.call_args.kwargs["fixed_coefficients"] == {0: 0}
    np.testing.assert_array_equal(
        curve_fitter.call_args.kwargs["literature_energies_kev"], [100, 200]
    )
    for index, peak in enumerate([100.0, 200.0]):
        call = peak_fitter.call_args_list[index].kwargs
        assert call["model_name"] == gauss_on_step.name
        assert call["mode_guess"] == 2 * peak
        assert call["fit_range"] == (2 * peak - 20, 2 * peak + 20)
        np.testing.assert_array_equal(
            call["energies"], 2 * peak + np.array([-5, -2, 0])
        )
        posterior = calibration.bayesian_peak_posteriors[0][peak]
        assert calibration.bayesian_peak_fits[0][peak] is artifact
        np.testing.assert_array_equal(
            curve_fitter.call_args.kwargs["peak_mode_draws"][index],
            posterior[peak_param],
        )
        peak_result = result["peak_parameters"][peak]
        assert peak_result["validity"] is True
        assert peak_result["position"] == pytest.approx(np.mean(posterior[peak_param]))
        assert peak_result["position_uncertainty"] == pytest.approx(
            np.std(posterior[peak_param], ddof=1)
        )
        parameter_draws = np.column_stack(
            [posterior[name] for name in gauss_on_step.required_args()]
        )
        np.testing.assert_allclose(
            peak_result["parameters"], parameter_draws.mean(axis=0)
        )
        np.testing.assert_allclose(
            peak_result["covariance"], np.cov(parameter_draws, rowvar=False, ddof=1)
        )
        np.testing.assert_allclose(
            peak_result["uncertainties"], np.std(parameter_draws, axis=0, ddof=1)
        )
        assert np.isnan(peak_result["p_value"])
        assert np.isnan(peak_result["chi_square"]).all()
    assert set(result) == {
        "peak_parameters",
        "peak_param",
        "pk_pos",
        "pk_pos_uncertainties",
        *expected_curve,
    }
    frequentist.assert_not_called()


def test_peak_only_calls_preserve_calibration_and_history(calibration):
    artifact = object()
    curve_fitter = Mock()
    calibration.set_bayesian_fitters(
        lambda **kwargs: (peak_draws(kwargs["mode_guess"]), artifact), curve_fitter
    )
    original_pars = calibration.pars.copy()
    for _ in range(2):
        fit_peaks(calibration, update_cal_pars=False)
    np.testing.assert_array_equal(calibration.pars, original_pars)
    assert len(calibration.bayesian_peak_posteriors) == 2
    assert len(calibration.bayesian_peak_fits) == 2
    assert len(calibration.results) == 2
    assert calibration.bayesian_calibration_posteriors == []
    curve_fitter.assert_not_called()


def test_failed_peak_is_excluded_from_curve(calibration):
    def peak_fitter(**kwargs):
        if kwargs["mode_guess"] == 200:
            msg = "sampler failed"
            raise RuntimeError(msg)
        return peak_draws(kwargs["mode_guess"]), None

    curve_fitter = Mock(return_value=(curve_result(), None))
    calibration.set_bayesian_fitters(peak_fitter, curve_fitter)
    fit_peaks(calibration)
    result = calibration.results["hpge_fit_energy_peaks"]
    assert result["peak_parameters"][100.0]["validity"] is False
    assert result["peak_parameters"][200.0]["validity"] is True
    assert set(calibration.bayesian_peak_posteriors[0]) == {200.0}
    np.testing.assert_array_equal(
        curve_fitter.call_args.kwargs["literature_energies_kev"], [200]
    )
    assert len(curve_fitter.call_args.kwargs["peak_mode_draws"]) == 1


@pytest.mark.parametrize("error", [RuntimeError, NotImplementedError])
def test_peak_backend_failure_in_debug_mode(calibration, error):
    calibration.debug_mode = True
    calibration.set_bayesian_fitters(Mock(side_effect=error("sampler failed")), Mock())
    with pytest.raises(
        (RuntimeError, NotImplementedError), match=r"sampler failed|peak fit failed"
    ):
        fit_peaks(calibration)


@pytest.mark.parametrize(
    ("name", "bad_draws", "message"),
    [
        ("mode", None, "missing draws"),
        ("mu", [1], "one-dimensional"),
        ("mu", [[1, 2]], "one-dimensional"),
        ("mu", [1, 2], "equal length"),
        ("mu", np.full(8, np.nan), "non-finite"),
        ("mu", ["invalid"] * 8, "numeric"),
    ],
)
def test_invalid_peak_draws_raise_contract_error(calibration, name, bad_draws, message):
    draws = peak_draws()
    if bad_draws is None:
        del draws[name]
    else:
        draws[name] = bad_draws
    calibration.set_bayesian_fitters(Mock(return_value=(draws, None)), Mock())
    with pytest.raises(energy_cal.BayesianFitBackendError, match=message):
        fit_peaks(calibration)
    assert calibration.bayesian_peak_posteriors == []


@pytest.mark.parametrize("output", [None, {}, ({},), ([], None)])
def test_invalid_peak_backend_payload(calibration, output):
    calibration.set_bayesian_fitters(Mock(return_value=output), Mock())
    with pytest.raises(energy_cal.BayesianFitBackendError):
        fit_peaks(calibration)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("extra_key", [1, 2], "topology"),
        ("pk_cal_pars", [1], "shape"),
        ("pk_cal_errs", [1], "shape"),
        ("calibration_covariance", [[1]], "shape"),
        ("calibration_parameters", [0, np.inf], "non-finite"),
        ("calibration_parameters", ["invalid", 0], "numeric"),
    ],
)
def test_invalid_curve_result_does_not_change_calibration(
    calibration, key, value, message
):
    result = curve_result()
    result[key] = value
    original_pars = calibration.pars.copy()
    calibration.set_bayesian_fitters(
        lambda **kwargs: (peak_draws(kwargs["mode_guess"]), None),
        Mock(return_value=(result, object())),
    )
    with pytest.raises(energy_cal.BayesianFitBackendError, match=message):
        fit_peaks(calibration)
    np.testing.assert_array_equal(calibration.pars, original_pars)
    assert calibration.bayesian_calibration_posteriors == []


def test_curve_backend_failure_is_explicit(calibration):
    calibration.set_bayesian_fitters(
        lambda **kwargs: (peak_draws(kwargs["mode_guess"]), None),
        Mock(side_effect=RuntimeError("sampler failed")),
    )
    with pytest.raises(
        energy_cal.BayesianFitBackendError, match="calibration fitter failed"
    ):
        fit_peaks(calibration)
    assert calibration.bayesian_calibration_posteriors == []


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_curve_respects_polynomial_degree(degree):
    cal = energy_cal.HPGeCalibration("energy", [100, 200, 300], 0.5, deg=degree)
    expected = curve_result(size=len(cal.pars))
    fitter = Mock(return_value=(expected, None))
    cal.set_bayesian_fitters(Mock(), fitter)
    result = cal._fit_calibration_curve(
        np.array([200, 400, 600]),
        np.ones(3),
        np.array([100, 200, 300]),
        peak_mode_draws=[peak_draws()["mode"]] * 3,
    )
    assert fitter.call_args.kwargs["degree"] == degree
    assert result["calibration_parameters"].shape == cal.pars.shape


def test_disabling_fitters_restores_frequentist_curve(calibration):
    calibration.set_bayesian_fitters(Mock(), Mock())
    calibration.set_bayesian_fitters(None, None)
    positions = np.array([200.0, 400.0])
    variances = np.array([0.01, 0.04])
    energies = np.array([100.0, 200.0])
    scale = energy_cal.hpge_fit_energy_scale(
        positions, variances, energies, deg=0, fixed=calibration.fixed
    )
    curve = energy_cal.hpge_fit_energy_cal_func(
        positions, variances, energies, scale[0], deg=0, fixed=calibration.fixed
    )
    result = calibration._fit_calibration_curve(positions, variances, energies)
    for key, expected in zip(
        [
            "pk_cal_pars",
            "pk_cal_errs",
            "pk_cal_cov",
            "calibration_parameters",
            "calibration_uncertainties",
            "calibration_covariance",
        ],
        (*scale, *curve),
        strict=True,
    ):
        np.testing.assert_allclose(result[key], expected)
    assert calibration.bayesian_peak_posteriors == []
    assert calibration.bayesian_calibration_posteriors == []
