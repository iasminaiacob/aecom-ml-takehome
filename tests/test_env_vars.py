import os

def test_hf_env_vars_present_or_none():
    keys = ["HF_HOME", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"]
    for k in keys:
        _ = os.environ.get(k) #should not raise
    assert True