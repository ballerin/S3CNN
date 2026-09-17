"""The scheduled-job runner and the DataLoader policy.

`run_all_experiments` claims one GPU and spends it on one job at a time. What
is worth pinning is the queueing, not the training: the jobs here are cheap
stand-ins passed through the same `run_schedule` seam the real entry point
uses, so the subprocess machinery is exercised for real.

The `utils.data_utils` tests pin the decision, not the measurement behind it;
that lives in `bench/data.py`.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import run_all_experiments as runner
from utils import data_utils


def _trace(params):
    """Stand-in job: records its window, then obeys `fail`."""
    with open(params["trace"], "a") as fh:
        fh.write(f"start {params['log_id']}\n")
        time.sleep(0.2)
        fh.write(f"end {params['log_id']}\n")
    if params.get("fail"):
        raise RuntimeError("job asked to fail")


def _report_device(params):
    with open(params["trace"], "a") as fh:
        fh.write(os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>") + "\n")


def _schedule(tmp_path, n, failing=()):
    trace = tmp_path / "trace"
    return trace, [{"operation": "train", "log_id": i, "trace": str(trace),
                    "fail": i in failing} for i in range(n)]


def test_jobs_do_not_overlap(tmp_path):
    """One card, one job: a second job must not start before the first ends."""
    trace, tasks = _schedule(tmp_path, 3)
    runner.run_schedule(tasks, target=_trace)
    assert trace.read_text().split("\n")[:6] == [
        "start 0", "end 0", "start 1", "end 1", "start 2", "end 2"]


def test_a_failing_job_does_not_stop_the_queue(tmp_path):
    """An overnight schedule should survive one bad set of parameters."""
    trace, tasks = _schedule(tmp_path, 3, failing={1})
    results = runner.run_schedule(tasks, target=_trace)
    codes = [code for _, code, _ in results]
    assert codes[0] == 0 and codes[1] != 0 and codes[2] == 0
    assert "start 2" in trace.read_text()


def test_the_claimed_device_reaches_the_job(tmp_path):
    """The parent claims the GPU; the child inherits the pin and sees cuda:0."""
    trace, tasks = _schedule(tmp_path, 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-test-uuid"
    try:
        runner.run_schedule(tasks, target=_report_device)
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    assert trace.read_text().strip() == "GPU-test-uuid"


@pytest.mark.parametrize("op", [None, "trian", "", 0])
def test_validate_rejects_an_unknown_operation(op):
    """A typo must be caught before a device is claimed, not after."""
    with pytest.raises(ValueError, match="operation"):
        runner.validate([{"operation": "train"}, {"operation": op}])


def test_describe_skips_keys_the_job_does_not_set():
    assert runner.describe({"operation": "train", "model_type": "S3CNNUNet"}) \
        == "train S3CNNUNet"


def test_summarise_counts_the_failures():
    out = runner.summarise([("a", 0, 60.0), ("b", 1, 30.0)])
    assert "FAILED rc=1" in out
    assert "2 job(s), 1 failed" in out


class _Fake(torch.utils.data.Dataset):
    """A dataset that can claim either regime."""

    def __init__(self, preloaded: bool, n: int = 8):
        self.is_preloaded = preloaded
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.zeros(2), 0


def test_a_preloaded_dataset_gets_no_workers():
    """`__getitem__` is a list index; a worker can only pickle it back slower."""
    assert data_utils.worker_count(_Fake(True)) == 0


def test_a_dataset_that_does_work_per_item_gets_workers():
    assert data_utils.worker_count(_Fake(False)) == data_utils.MAX_WORKERS


def test_a_dataset_with_no_opinion_is_treated_as_doing_work():
    """`is_preloaded` is set by SphericalDataset; anything else is not free."""
    assert data_utils.worker_count(object()) == data_utils.MAX_WORKERS


def test_the_affinity_mask_caps_the_count(monkeypatch):
    """A batch system hands out cores; oversubscribing them is someone else's problem."""
    monkeypatch.setattr(data_utils, "available_cpus", lambda: 2)
    assert data_utils.worker_count(_Fake(False)) == 1
    monkeypatch.setattr(data_utils, "available_cpus", lambda: 1)
    assert data_utils.worker_count(_Fake(False), requested=8) == 0


def test_make_dataloader_pins_and_keeps_workers_alive():
    loader = data_utils.make_dataloader(_Fake(False), batch_size=2, shuffle=False)
    assert loader.pin_memory
    assert loader.num_workers == data_utils.MAX_WORKERS
    assert loader.persistent_workers


def test_make_dataloader_does_not_ask_for_persistence_it_cannot_have():
    """`persistent_workers=True` with no workers is a DataLoader ValueError."""
    loader = data_utils.make_dataloader(_Fake(True), batch_size=2, shuffle=False)
    assert loader.num_workers == 0
    assert not loader.persistent_workers
    assert [len(x) for x, _ in loader] == [2, 2, 2, 2]


# ------------------------------------------------------- schedule parameters

def _job(**overrides):
    job = {"operation": "train", "experiment": "wind->wind",
           "model_type": "S3CNNUNet", "rotate": "no", "batch_size": 1,
           "total_epochs": 1}
    job.update(overrides)
    return job


def test_a_well_formed_job_passes():
    runner.validate([_job()])


@pytest.mark.parametrize("typo, meant", [
    ("pretrained_wights", "pretrained_weights"),   # shipped in the template
    ("batch_norm", "use_batch_norm"),              # shipped in the template
    ("verbos", "verbose"),
])
def test_a_misspelled_key_is_rejected_and_the_right_one_suggested(typo, meant):
    """`params.get` turns a typo into a silent default, which is how both of
    the template's own misspellings survived."""
    with pytest.raises(ValueError, match=f"did you mean '{meant}'"):
        runner.validate([_job(**{typo: True})])


def test_the_failing_job_is_named():
    with pytest.raises(ValueError, match="job 2:"):
        runner.validate([_job(), _job(verbos=True)])


def test_keys_are_checked_against_the_operations_own_entry_point():
    """`path` is required by `run_testing` and unknown to `run_training`."""
    test_job = _job(operation="test", path="somewhere", nonlinearity=None,
                    use_batch_norm=False, use_layer_norm=False)
    del test_job["total_epochs"]        # trains for epochs; testing does not
    runner.validate([test_job])
    with pytest.raises(ValueError, match="'path'"):
        runner.validate([_job(path="somewhere")])


def test_the_shipped_template_validates():
    """The template is what a new user copies; it should not need fixing."""
    import TEMPLATE_train_schedule
    runner.validate(TEMPLATE_train_schedule.create_schedule())
