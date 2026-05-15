# Reproduce the training artifacts

This is the build script the project owner used to produce the artifacts in `public/data/`. Run it only if you want to regenerate artifacts from scratch — they're already committed in the repo, so the site works without re-running this.

## Kaggle, ~7 hours on a single T4

```python
# 1. clone (на Kaggle %cd сохраняется между cells, !cd — нет)
!git clone https://github.com/personalitytg/pretraining-replay
%cd pretraining-replay

# 2. install
!pip install -q -r requirements.txt

# 3. prepare data (~30 min, downloads 3 GB TinyStories)
!python scripts/prepare_data.py

# 4. secrets из Kaggle Add-ons → Secrets (WANDB_API_KEY должен быть attached)
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")

# 5. train (~5-7 hours on T4)
!python scripts/train.py --config scripts/configs/tinystories_30m.yaml

# 6. derive frontend artifacts (~1 hour for generate, ~5 min for hashing)
!python scripts/generate_artifacts.py --runs-dir runs/tinystories_30m_v1 --out-dir public/data --device cuda
!python scripts/build_manifest.py --runs-dir runs/tinystories_30m_v1 --out-dir public/data
!python scripts/detect_discoveries.py --out-dir public/data
!python scripts/verify_artifacts.py --data-dir public/data

# 7. zip frontend artifacts for local import
import shutil
shutil.make_archive('/kaggle/working/artifacts', 'zip', 'public/data')
```
