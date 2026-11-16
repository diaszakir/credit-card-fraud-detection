"""Restore verified report seeds into Docker volumes without reading test."""
import json

import pytest

import app.prepare as prepare


def test_evaluation_seed_populates_old_volume_and_refuses_conflict(data_project, monkeypatch):
    from src.evaluation.final_test import evaluate_test
    root = data_project['root']
    seed = root / 'image_seed'
    seed.mkdir()
    document = evaluate_test(root, allow_non_real=True)['result']
    (seed / 'test_final.json').write_text(json.dumps(document))
    (seed / 'permutation_importance_validation.csv').write_text('feature,importance\nV1,0.1\n')
    monkeypatch.setattr(prepare, 'ROOT_DIR', root)
    monkeypatch.setattr(prepare, 'EVALUATION_SEED', seed)
    # Startup must not reopen test to display results from the image.
    import src.data.api
    monkeypatch.setattr(src.data.api, 'load_xy', lambda *a, **k: pytest.fail('test must not be opened during seed restore'))
    prepare.restore_evaluation_reports()
    target = root / 'reports/experiments/evaluation/test_final.json'
    assert json.loads(target.read_text()) == json.loads((seed / 'test_final.json').read_text())
    prepare.restore_evaluation_reports()  # repeat is safe
    document['fingerprint']['threshold'] = 0.99
    target.write_text(json.dumps(document))
    previous = target.read_bytes()
    with pytest.raises(ValueError, match='другую итоговую'):
        prepare.restore_evaluation_reports()
    assert target.read_bytes() == previous


def test_evaluation_seed_upgrades_only_preserved_legacy(data_project, monkeypatch):
    from src.evaluation.final_test import evaluate_test
    root = data_project['root']
    document = evaluate_test(root, allow_non_real=True)['result']
    legacy = json.loads(json.dumps(document))
    legacy.pop('model_identity')
    legacy['fingerprint']['artifact_sha256'] = '0' * 64
    document['provenance_verification'] = {'original_result': legacy}
    seed = root / 'image_seed'
    seed.mkdir()
    (seed / 'test_final.json').write_text(json.dumps(document))
    target = root / 'reports/experiments/evaluation/test_final.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(legacy))
    monkeypatch.setattr(prepare, 'ROOT_DIR', root)
    monkeypatch.setattr(prepare, 'EVALUATION_SEED', seed)
    prepare.restore_evaluation_reports()
    assert json.loads(target.read_text()) == json.loads((seed / 'test_final.json').read_text())
