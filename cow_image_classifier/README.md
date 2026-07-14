# CowDocAI — Image Classification Module

EfficientNet-B0 transfer learning for cattle disease detection from images,
matching the architecture: preprocess → EfficientNet-B0 → frozen early layers
→ fine-tuned last layers → classifier → softmax → predicted disease + confidence.

## 1. Expected folder structure

```
your_project/
  dataset/
    train/
      LSD/
      FMD/
      Mastitis/
      Ringworm/
      BovinePapillomatosis/
      ... (one folder per class, matching your organized data)
    val/
      LSD/
      FMD/
      ...
```

If your images are currently in per-class folders **without** a train/val split
(e.g. `raw_dataset/LSD/*.jpg`), run `split_dataset.py` once to create the split
above (edit `SOURCE_DIR` at the top of that file first).

Class names are read automatically from the folder names — nothing to hardcode
in the code itself.

## 2. Install

```bash
pip install -r requirements.txt
```

Check whether PyTorch sees a GPU (optional — CPU works fine for EfficientNet-B0
on a small dataset, just slower):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 3. Train

```bash
python train.py
```

This runs two phases automatically:
- **Phase 1**: backbone frozen, only the new classifier head trains (fast).
- **Phase 2**: last few stages unfrozen, fine-tuned at a low learning rate.

Best checkpoint (by validation accuracy) is saved to `checkpoints/best_model.pt`,
along with `checkpoints/class_names.json`. At the end you'll get a full
per-class precision/recall/F1 report — check this closely, not just overall
accuracy, since some of your classes (e.g. Ringworm, Bovine Papillomatosis)
likely have far fewer images than others (FMD/LSD via Zenodo), and a
high-consequence disease like FMD being under-recalled matters more than the
raw accuracy number would suggest.

## 4. Predict

```bash
python predict.py path/to/cow_photo.jpg
```

Or import `predict_disease()` directly in your Streamlit app / conversation
manager — it returns the top-k diseases with confidence scores, which you can
feed straight into your existing disease-filtered RAG stage the same way you
already do with the LLM-identified disease from the text pipeline.

## Notes / things worth double-checking

- **Class list alignment**: your text pipeline's disease list (LSD, FMD,
  Mastitis, Foot Rot, Ringworm) and your image classifier's classes may not
  match exactly — e.g. there's no public Foot Rot image dataset, so it's
  likely absent from your image classes unless you collected your own. Worth
  deciding explicitly whether the image path should ever predict a disease
  the text/RAG side doesn't know about, or vice versa.
- **Class imbalance** is already handled via a weighted sampler in
  `dataset.py`, but if a class has very few images (say, under ~30), consider
  collecting more before trusting its recall.
- **Next step (optional)**: Grad-CAM to visualize which region of the image
  drove a prediction — genuinely useful if you're getting vet input from
  Justin Davis at KVASU, since it lets him sanity-check *why* the model
  flagged a disease, not just what it predicted.