# Continual forecasting baselines

These adapters run the full-data, chronological 60/20/20 protocol with input and forecast windows of 12 and source-scale inputs. Each method uses its own adaptation rule and loss. For instance, TrafficStream updates a subgraph with Fisher regularization, and STBP updates the pattern bank after the first stage. Validation selects the checkpoint evaluated at each stage.

Install a compatible PyTorch build, then run from the repository root:

```bash
python -m pip install -r continual/requirements.txt
python continual/install_upstream.py all
bash run_baseline.sh STBP PEMS 0 42
bash run_baseline.sh all all 0 42
```

The launcher accepts one method or `all` and one dataset or `all`. It writes timestamped outputs under `runs/` and reads the dataset archive extracted to `data/`. Override training options after the seed, for example `--epochs 1 --num-workers 0`. Madrid has a separate observation mask that preserves observed zero traffic for MAE and RMSE. The original Madrid runners cover TS, STKEC, PECPM, EAC, and STBP. The suite explicitly skips STRAP on Madrid. An individual request for that combination exits with an error.

The launcher sets 200 epochs and a patience of 30. The archived full-data settings use batch 64 and a learning rate of 0.01 for STBP on PEMS and AIR, and batch 128 and a learning rate of 0.005 for the remaining measured combinations. STRAP uses the latter training budget for new runs. An epoch-limited smoke run does not reproduce a converged score.

## Upstream implementations

| Method | Authors' source | Integration |
| --- | --- | --- |
| TrafficStream (TS) | [AprLie/TrafficStream](https://github.com/AprLie/TrafficStream) | Native predictor, changes and replay candidates selected from training data, Fisher regularization |
| STKEC | [UnderReview24/STKEC](https://github.com/UnderReview24/STKEC) | Native predictor with reconstructed clustering supervision and training-only preprocessing |
| PECPM | [UnderReview24/PECPM](https://github.com/UnderReview24/PECPM) | Native predictor with daily patterns, cluster centers, and auxiliary matching targets reconstructed from training data |
| EAC | [Onedean/EAC](https://github.com/Onedean/EAC) | Low-rank node prompts and original prompt-tuning mechanism |
| STRAP | [HoweyZ/STRAP](https://github.com/HoweyZ/STRAP) | Three pattern libraries built using training observations |
| STBP | [Aoyu-Liu/STBP](https://github.com/Aoyu-Liu/STBP) | Native predictor with pattern-only incremental tuning |

The upstream snapshots used for these integrations contain no repository-level license. The release contains the project-specific adapters, an upstream revision lock and checksums. `install_upstream.py` obtains the predictors directly from their authors and verifies their content after normalizing line endings. Where necessary it then converts the original GCN import to a package-relative import. A user with a matching local checkout can instead run `python continual/install_upstream.py STBP --checkout /path/to/STBP`. Upstream copyrights and terms apply to the downloaded predictors. The Madrid adapters preserve node identities, active-node masks and the selected month registry between stages.

