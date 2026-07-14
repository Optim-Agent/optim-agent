import pytest


def test_landing_animation_hpo_contract():
    from examples import rl_control as rl

    assert rl.LANDING_HPO_SEED == 0
    assert rl.LANDING_EVAL_EPISODES == 20
    assert rl.LANDING_MAX_TRIALS == 100
    assert rl._is_successful_landing(True, False, 100.0)
    assert not rl._is_successful_landing(True, False, -100.0)
    with pytest.raises(ValueError, match="at most 100"):
        rl._validate_landing_budget(101)


def test_reference_landing_controller_lands():
    from examples import rl_control as rl

    pytest.importorskip("gymnasium")
    episode = rl._evaluate_landing_params(rl.LANDING_REFERENCE_PARAMS, (0,))[0]

    assert episode["landed"] is True
    assert episode["final_signal"] == 100.0
