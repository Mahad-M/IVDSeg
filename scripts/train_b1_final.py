"""Final train-only B1 ResUNet retraining."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from ivdseg.training import verify_cuda_runtime
from ivdseg.unet_training import B1FinalTrainingConfig, build_b1_final_training

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="B1-resunet34-final"); parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--normalization-profile", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--accelerator", default="gpu"); parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    config = B1FinalTrainingConfig(manifest_path=args.manifest, dataset_root=args.dataset_root, normalization_profile=args.normalization_profile,
        run_dir=args.run_dir, experiment_id=args.experiment_id, seed=args.seed, accelerator=args.accelerator, num_workers=args.num_workers)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(json.dumps({"schema_version":1,"experiment_id":config.experiment_id,"b1_final_training":{**config.__dict__,"manifest_path":str(config.manifest_path),"dataset_root":str(config.dataset_root),"normalization_profile":str(config.normalization_profile),"run_dir":str(config.run_dir),"train_subject_ids":["01","02","04","05","06","08","09","11","12","13","15","16"],"validation_subject_ids":[],"fixed_test_subjects_excluded":["03","07","10","14"],"effective_batch_size":config.effective_batch_size,"initialization":"from_scratch"}},indent=2)+"\n")
    verify_cuda_runtime(); trainer,module,data_module,_checkpoint=build_b1_final_training(config); trainer.fit(module,datamodule=data_module)
if __name__ == "__main__": main()
