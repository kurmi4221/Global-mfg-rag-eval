import json

import pytest

from mfg_rag_eval.ledger import RunLedger, verify_ledger


def test_chain_verifies_and_resumes(tmp_path):
    path = tmp_path / "l.jsonl"
    ledger = RunLedger(path)
    for i in range(5):
        ledger.append({"i": i, "text": "caf\u00e9"})
    assert verify_ledger(path)[0]
    resumed = RunLedger(path)
    resumed.append({"i": 5})
    assert verify_ledger(path)[0]


def test_modified_record_detected(tmp_path):
    path = tmp_path / "l.jsonl"
    ledger = RunLedger(path)
    for i in range(3):
        ledger.append({"score": i})
    lines = path.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["record"]["score"] = 99
    lines[1] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n")
    ok, msg = verify_ledger(path)
    assert not ok and "modified" in msg


def test_deleted_or_reordered_line_detected(tmp_path):
    path = tmp_path / "l.jsonl"
    ledger = RunLedger(path)
    for i in range(4):
        ledger.append({"n": i})
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2], lines[3]]) + "\n")
    assert not verify_ledger(path)[0]


def test_refuses_to_append_to_corrupt_ledger(tmp_path):
    path = tmp_path / "l.jsonl"
    RunLedger(path).append({"a": 1})
    path.write_text(path.read_text().replace('"a":1', '"a":2'))
    with pytest.raises(ValueError):
        RunLedger(path)


def test_missing_file_is_invalid(tmp_path):
    assert not verify_ledger(tmp_path / "nope.jsonl")[0]
