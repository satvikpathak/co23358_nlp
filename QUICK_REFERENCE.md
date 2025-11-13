# 🚀 Quick Reference Card - Tone Transformer

## Essential Commands

### Start Everything (One Command!)
```bash
./run.sh
```

### Other Commands
```bash
./run.sh status    # Check services
./run.sh stop      # Stop all
./run.sh restart   # Restart all
./run.sh logs      # View logs
```

---

## URLs

| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:8080 |
| **Backend API** | http://localhost:8001 |
| **API Docs** | http://localhost:8001/docs |
| **TensorBoard** | http://localhost:6006 |

---

## Training Workflow

1. **Prepare Data** (5 min)
   ```bash
   python prepare_dataset.py --source synthetic --size large
   ```

2. **Train Model** (30-60 min)
   ```bash
   jupyter notebook train_model.ipynb
   # Run all cells
   ```

3. **Evaluate** (5 min)
   ```bash
   jupyter notebook evaluate_model.ipynb
   # Run all cells
   ```

4. **Deploy**
   ```bash
   ./run.sh
   ```

---

## File Structure

```
nlp/
├── run.sh                  # Start script
├── train_model.ipynb       # Training (100 epochs)
├── evaluate_model.ipynb    # Evaluation metrics
├── prepare_dataset.py      # Data preparation
├── backend/                # FastAPI server
├── frontend/               # Web interface
└── models/                 # Trained models
```

---

## API Testing

```bash
curl -X POST "http://localhost:8001/rewrite" \
  -H "Content-Type: application/json" \
  -d '{"text":"need this asap", "tone":"formal"}'
```

---

## Evaluation Metrics

| Metric | Good Score | Your Score |
|--------|------------|------------|
| BLEU | > 30 | Run evaluate_model.ipynb |
| ROUGE-1 | > 0.5 | See metrics_summary.json |
| ROUGE-2 | > 0.3 | Check evaluation output |
| Perplexity | < 50 | Lower is better |

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Port in use | `./run.sh stop` then `./run.sh start` |
| Out of memory | Reduce batch_size in notebook |
| Model not found | Train first or use pretrained |
| Dependencies fail | `pip install --upgrade pip` |

---

## Dataset Options

```bash
# Synthetic (default)
python prepare_dataset.py --source synthetic --size large

# Hugging Face
python prepare_dataset.py --source huggingface

# Custom JSON
python prepare_dataset.py --source custom --input data.json
```

---

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Epochs | 100 | Training iterations |
| Batch size | 8 | Samples per batch |
| Learning rate | 5e-5 | Optimization rate |
| Model | flan-t5-base | 220M parameters |
| Max length | 128 | Token limit |

---

## Performance Tips

**Faster Training:**
- Use GPU
- Enable FP16
- Use smaller model (flan-t5-small)
- Reduce epochs to 50

**Better Accuracy:**
- Use larger dataset
- Train full 100 epochs
- Use flan-t5-base or larger
- Increase beam search width

---

## Example Transformations

| Informal | Formal |
|----------|--------|
| send me the file now | Could you please send me the file at your earliest convenience? |
| need this asap | I would appreciate receiving this as soon as possible. |
| why didn't you finish? | I noticed the task is incomplete. Could you provide an update? |
| good job | Excellent work on this project. |
| i dunno | I am uncertain about that. |

---

## Project Outputs

After running everything:
- ✓ Trained model in `models/`
- ✓ Evaluation metrics in `metrics_summary.json`
- ✓ Detailed results in `evaluation_results.csv`
- ✓ Visualizations as PNG files
- ✓ Training logs in TensorBoard

---

## Next Steps

1. ✅ Run `./run.sh` to start
2. ✅ Test with examples
3. ✅ Train model (optional)
4. ✅ Evaluate performance
5. ✅ Use in production!

---

**Need detailed help?** → See README.md or SETUP.md
