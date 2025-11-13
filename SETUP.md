# Quick Setup Guide - Tone Transformer

## 🚀 One-Command Start (Easiest!)

```bash
./run.sh
```

That's it! The script will:
- ✓ Create virtual environment
- ✓ Install dependencies  
- ✓ Start backend (port 8001)
- ✓ Start frontend (port 8080)

**Access your app:**
- Frontend: http://localhost:8080
- Backend API: http://localhost:8001/docs

---

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB+ recommended)
- Internet connection (first run only)

---

## 📚 Step-by-Step Guide

### Step 1: Prepare Dataset (Optional - synthetic dataset included)

```bash
# Create a large synthetic dataset
python prepare_dataset.py --source synthetic --size large --output ./data/informal_formal
```

**Or use a real dataset:**
```bash
# From Hugging Face
python prepare_dataset.py --source huggingface --dataset jxm/informal_to_formal --output ./data/informal_formal
```

### Step 2: Train Model (100 Epochs)

```bash
# Start Jupyter
jupyter notebook train_model.ipynb
```

Then run all cells in the notebook. Training will:
- Load the dataset
- Fine-tune T5/FLAN-T5 for 100 epochs
- Save model to `./models/informal-to-formal-t5/`
- Log metrics to TensorBoard

**Monitor training:**
```bash
tensorboard --logdir ./models/informal-to-formal-t5/logs
```

**Training time:**
- CPU: 2-4 hours (small model)
- GPU: 30-60 minutes (base model)

### Step 3: Evaluate Model

```bash
# Open evaluation notebook
jupyter notebook evaluate_model.ipynb
```

Run all cells to get:
- BLEU score
- ROUGE scores  
- Perplexity
- Visualizations
- Detailed metrics

### Step 4: Run Application

```bash
# Start both frontend and backend
./run.sh
```

**Other commands:**
```bash
./run.sh status   # Check if running
./run.sh stop     # Stop services
./run.sh restart  # Restart everything
./run.sh logs     # View logs
```

---

## 🎯 Without Training (Use Pretrained Model)

Don't want to train? No problem!

```bash
./run.sh
```

The app will automatically use the pretrained `flan-t5-small` model. It works well but a fine-tuned model will be more accurate.

---

## 🧪 Testing the API

```bash
# Test the API
curl -X POST "http://localhost:8001/rewrite" \
  -H "Content-Type: application/json" \
  -d '{"text":"send me the file now", "tone":"formal"}'
```

Expected response:
```json
{
  "original": "send me the file now",
  "rewritten": "Could you please send me the file at your earliest convenience?",
  "tone": "formal"
}
```

---

## 📊 Project Workflow

```
1. Prepare Data → 2. Train Model → 3. Evaluate → 4. Deploy → 5. Use!
     (5 min)          (30-60 min)     (5 min)     (1 min)
```

**Shortcut:** Skip steps 1-3 and use pretrained model!

---

## 🔧 Troubleshooting

### Port already in use?
```bash
./run.sh stop
./run.sh start
```

### Out of memory during training?
Edit `train_model.ipynb` and change:
- `per_device_train_batch_size=8` → `4`
- Or use `google/flan-t5-small` instead of `base`

### Dependencies not installing?
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Can't find jupyter?
```bash
pip install jupyter ipykernel
```

---

## 📁 What Gets Created

After running everything, you'll have:

```
nlp/
├── .venv/                       # Virtual environment
├── models/                      # Your trained model
│   └── informal-to-formal-t5/
├── data/                        # Training data
│   └── informal_formal/
├── evaluation_results.csv       # Metrics
├── metrics_summary.json         # Performance data
├── *.png                        # Visualizations
└── *.log                        # Service logs
```

---

## 🎓 For Academic Use

### Minimum Viable Demo (5 minutes)
```bash
./run.sh
# Open http://localhost:8080
# Demo the text transformation!
```

### Full Project with Training (1-2 hours)
1. Prepare dataset: `python prepare_dataset.py --source synthetic --size large`
2. Train model: Run `train_model.ipynb` (all cells)
3. Evaluate: Run `evaluate_model.ipynb` (all cells)
4. Deploy: `./run.sh`
5. Create report using metrics and visualizations

### What to Include in Report
- Problem statement & motivation
- Dataset description (source, size, examples)
- Model architecture (T5/FLAN-T5, parameters)
- Training configuration (100 epochs, batch size, etc.)
- Evaluation metrics (BLEU, ROUGE, perplexity)
- Example outputs (show transformations)
- Visualizations (from evaluate_model.ipynb)
- Discussion of results & future improvements

---

## 💡 Quick Tips

**Want faster training?**
- Use `flan-t5-small` (60M params) instead of `base` (220M)
- Reduce epochs from 100 to 50
- Use GPU if available

**Want better accuracy?**
- Use larger dataset (GYAFC or custom)
- Train for full 100 epochs
- Use `flan-t5-base` or `flan-t5-large`
- Increase batch size if you have GPU

**Need help?**
- Check README.md for detailed docs
- Review troubleshooting section
- Check logs: `./run.sh logs`

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Virtual environment created: `ls .venv/`
- [ ] Dependencies installed: `pip list | grep transformers`
- [ ] Dataset prepared: `ls data/informal_formal/`
- [ ] Model trained: `ls models/informal-to-formal-t5/`
- [ ] Backend running: `curl http://localhost:8001/docs`
- [ ] Frontend running: Open http://localhost:8080
- [ ] API works: Send test request
- [ ] Evaluation complete: Check `metrics_summary.json`

---

## 🎉 You're All Set!

Your Tone Transformer is ready to use. Try transforming some informal text to formal!

**Need help?** Check README.md for comprehensive documentation.
