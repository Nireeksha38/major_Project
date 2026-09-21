import pytest
from risk_engine.risk_classifier import RiskClassifier
from risk_engine.thresholds import RiskThresholds
from risk_engine.alert_logic import AlertManager


def test_risk_score_green_normal():
    classifier = RiskClassifier()

    eval_res = classifier.evaluate(
        people_count=12,
        density_score=0.20,
        relative_speed=0.15,
        motion_intensity=0.15,
        motion_entropy=0.20,
        panic_score=0.10,
        fall_detected=False,
        num_falls=0,
        fall_score=0.0
    )

    assert eval_res["risk_level"] == "GREEN"
    assert eval_res["status_title"] == "NORMAL"
    assert eval_res["risk_score"] < 0.39


def test_risk_score_red_critical_panic():
    classifier = RiskClassifier()

    eval_res = classifier.evaluate(
        people_count=45,
        density_score=0.85,
        relative_speed=0.90,
        motion_intensity=0.80,
        motion_entropy=0.85,
        panic_score=0.88,
        fall_detected=False,
        num_falls=0,
        fall_score=0.0
    )

    assert eval_res["risk_level"] == "RED"
    assert eval_res["status_title"] == "CRITICAL"
    assert eval_res["risk_score"] >= 0.70


def test_fall_detected_overrides_risk():
    classifier = RiskClassifier()

    eval_res = classifier.evaluate(
        people_count=15,
        density_score=0.45,
        relative_speed=0.20,
        motion_intensity=0.20,
        motion_entropy=0.20,
        panic_score=0.10,
        fall_detected=True,
        num_falls=1,
        fall_score=0.75
    )

    # In a crowd, a confirmed fall must immediately trigger at least WARNING/CRITICAL
    assert eval_res["risk_level"] in ["YELLOW", "RED"]
    assert "fall" in eval_res["summary_message"].lower()


def test_alert_manager_cooldown():
    alert_mgr = AlertManager(cooldown_seconds=5)

    risk_eval = {"risk_level": "RED", "risk_score": 0.85}
    panic_info = {"is_rush": False, "is_panic": True, "event": "PANIC_MOVEMENT"}
    fall_info = {"fall_detected": False, "num_falls": 0, "details": []}

    # Trigger 1: Should fire
    alerts_1 = alert_mgr.check_and_trigger(None, risk_eval, panic_info, fall_info)
    assert len(alerts_1) == 1

    # Immediate Trigger 2: Should be blocked by cooldown
    alerts_2 = alert_mgr.check_and_trigger(None, risk_eval, panic_info, fall_info)
    assert len(alerts_2) == 0
