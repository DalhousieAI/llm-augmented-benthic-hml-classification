# GHiML - A VLM-Guided Hierarchical Multi-Label Framework for Benthic Imagery Classification

Code for variable proposal, captioning, embedding and hierarchical multi-label training.

## Requirements

Python 3.10+, a CUDA GPU with flash-attention support, and the packages in `requirements.txt`.
Models used: `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-Embedding-8B`, DINOv3 ViT-H.
Models are loaded from the Hugging Face Hub (pass `--login_token` where required) or, with `--compute_cluster`, from `./models/<name>/snapshots/tensors/`.

## Data

Place datasets under `./datasets/`:

- `datasets/benthicnet/`: `benthicnet_labelled.csv`, `benthicnet_biota.csv`, `benthicnet_substrates.csv`, images under `images/`
- `datasets/fathomnet-2025/`: `image_annotations.csv`, `roi_annotations.csv`, `full_annotations.csv`, `train/index_to_taxon.json`, `train/taxon_to_index.json`
- `datasets/inat/`: `train_mini.csv`, `val.csv`, `inat_annotations.csv`

## Pipeline

Examples are for BN-Bio. Use `sub` / `bn_sub` for BN-Sub. FathomNet uses `01_query_fn.py` (`--task_type fn|fn_cap --image_type main`) and `02_train_hml_model_fn.py`; iNaturalist uses `01_query_inat.py` (`--task_type inat|inat_cap`), `00_complete_captions.py` and `02_train_hml_model_inat.py`.

1. Variable proposal

       python 01_query.py --task_type bn_bio --linux

2. Variable selection (EM decomposition, KL scoring, deduplication, per-node selection)

       python 00_select_variables.py --dataset benthicnet --domain bio

3. Captioning

       python 01_query.py --task_type bn_bio_cap --full_data --linux

4. Embeddings

       python 01_extract_img_emb.py --use_full_emb --linux
       python 01_extract_caption_emb.py --dataset_domain bio --use_full_emb --batch_size 32

5. Tar shards

       python 00_convert_shards_to_tar.py --dataset_domain bio

6. Training (per seed). Branches first, then the routers.

       python 02_train_hml_model.py --feature_set bn_full_bio --seed 42 --modality img --num_epochs 35 --val_ratio 0.0
       python 02_train_hml_model.py --feature_set bn_full_bio --seed 42 --modality txt --num_epochs 35 --val_ratio 0.0 --lr 0.00003
       python 02_train_hml_model.py --feature_set bn_full_bio --seed 42 --modality full --num_epochs 35 --val_ratio 0.0
       python 02_train_hml_model.py --feature_set bn_full_bio --seed 42 --modality full --num_epochs 35 --val_ratio 0.0 --adaptive_fusion --lr 0.0001 --freeze_pooling
       python 02_train_hml_model.py --feature_set bn_full_bio --seed 42 --modality full --num_epochs 35 --val_ratio 0.0 --adaptive_fusion --lr 0.0001 --reinit_all

   The fourth command is Router and the fifth Router†. Add `--random_test_set` for the in-domain split.
