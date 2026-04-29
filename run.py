from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# 选择一个预训练配置（示例用 pi0.5 droid）
cfg = _config.get_config("pi05_droid")
ckpt_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 创建已训练策略
policy = policy_config.create_trained_policy(cfg, ckpt_dir)

# 先打印模型对象，确认已成功加载
print("Policy loaded:", type(policy))