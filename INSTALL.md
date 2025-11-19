# Installation Guide - Emergency NLP Classifier

## 🔧 Quick Fix for Keras 3 Error

If you're getting this error:
```
ValueError: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers.
```

### Solution:

Run these commands in your terminal:

```bash
# Install tf-keras (Keras 2 compatibility layer)
pip install tf-keras==2.16.0

# OR reinstall all requirements
pip install -r requirements.txt
```

---

## 📦 Complete Installation Steps

### Step 1: Clone Repository
```bash
git clone https://github.com/dushyantzz/emergency-nlp-classifier-sih2025.git
cd emergency-nlp-classifier-sih2025
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; import transformers; print('✅ All packages installed successfully')"
```

---

## 🚀 Training the Model

### Option 1: Use the Dataset from GitHub
```bash
python models/train_model.py
```

### Option 2: Generate New Dataset
```bash
python data/generate_dataset.py
python models/train_model.py
```

---

## 🔍 Troubleshooting

### Error: "No module named 'tf_keras'"
**Solution:**
```bash
pip install tf-keras==2.16.0
```

### Error: "Dataset not found"
**Solution:**
- Make sure `data/emergency_dataset.csv` exists
- If not, run: `python data/generate_dataset.py`

### Error: "CUDA out of memory" (GPU)
**Solution:**
- Reduce batch_size in `models/train_model.py` (line 19)
- Change from 16 to 8 or 4

### Error: "Slow training on CPU"
**Solution:**
- Use Google Colab with free GPU
- Or reduce dataset size for testing

---

## ⚡ Google Colab Setup (FREE GPU)

If you don't have a GPU, use Google Colab:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create new notebook
3. Enable GPU: Runtime → Change runtime type → GPU → Save
4. Run:

```python
!git clone https://github.com/dushyantzz/emergency-nlp-classifier-sih2025.git
%cd emergency-nlp-classifier-sih2025
!pip install -r requirements.txt
!python models/train_model.py
```

5. Download trained model:
```python
from google.colab import files
files.download('outputs/trained_model/tf_model.h5')
```

---

## 📋 System Requirements

### Minimum:
- Python 3.8+
- 8GB RAM
- 5GB free disk space
- CPU (training will be slow)

### Recommended:
- Python 3.10+
- 16GB RAM
- 10GB free disk space
- NVIDIA GPU with 6GB+ VRAM
- CUDA 11.8+

---

## ✅ Verify Everything Works

```bash
# Check Python version
python --version

# Check TensorFlow
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

# Check Transformers
python -c "import transformers; print('Transformers:', transformers.__version__)"

# Check tf-keras
python -c "import tf_keras; print('tf-keras installed ✅')"
```

---

## 🆘 Still Having Issues?

1. **Delete virtual environment and recreate:**
   ```bash
   rm -rf venv  # or rmdir /s venv on Windows
   python -m venv venv
   venv\Scripts\activate  # or source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Use Python 3.10 or 3.11** (most stable for this project)

4. **Check GitHub Issues** or create a new one

---

## 📚 Next Steps

After successful installation:

1. ✅ Train model: `python models/train_model.py`
2. ✅ Convert to TFLite: `python models/convert_to_tflite.py`
3. ✅ Test model: `python models/test_model.py`
4. ✅ Deploy to Android app

---

**Built for SIH 2025 - Team Daredevils** 🏆