---
license: "cc-by-4.0"
language:
  - ar
task_categories:
  - text-classification
tags:
  - arabic
  - dialect-identification
  - multilingual
  - sociolinguistics
  - code-switching
  - bivalency
configs:
  - config_name: full_text
    description: "Raw dialectal text files for EGY, GLF, LAV, NOR and MSA."
  - config_name: freq_lists
    description: "Dialect-specific vocabulary lists, bivalency-removed lists and MSA shared lists."
---

# Arabic Dialects Dataset (Bivalency & Code-Switching)

The **Arabic Dialects Dataset** is a specialised corpus designed for automatic dialect identification, with a focus on the linguistic phenomena of **bivalency** and **written code-switching** between major Arabic dialects and Modern Standard Arabic (MSA).  
It covers five varieties:

- **EGY** – Egyptian Arabic  
- **GLF** – Gulf Arabic  
- **LAV** – Levantine Arabic  
- **NOR** – North African / Tunisian Arabic  
- **MSA** – Modern Standard Arabic  

The dataset was created for research on fine-grained linguistic variation and has been used to evaluate new methods such as **Subtractive Bivalency Profiling (SBP)**, achieving over **76% accuracy** in supervised dialect identification.

This HuggingFace release makes the dataset machine-readable and ready for text classification, feature engineering, or corpus linguistic exploration.

---

## 📘 Citation

If you use this dataset, please cite:

**El-Haj M., Rayson P., Aboelezz M. (2018)**  
*Arabic Dialect Identification in the Context of Bivalency and Code-Switching.*  
In **Proceedings of the 11th International Conference on Language Resources and Evaluation (LREC 2018)**, Miyazaki, Japan, pp. 3622–3627.  
European Language Resources Association (ELRA).  
PDF: https://elhaj.uk/docs/237_Paper.pdf

---

## 📂 Dataset Structure

The dataset is distributed across two main sections:

### **1. Dialects Full Text**
Five files, each containing all instances belonging to one dialect:

```
Dialects Full Text/
│── allEGY.txt
│── allGLF.txt
│── allLAV.txt
│── allMSA.txt
└── allNOR.txt
```

---


Each file contains raw text samples, one per line, suitable for direct use in dialect classification experiments.

---

### **2. Dialectal Frequency Lists**

These resources support linguistic analysis and the SBP approach introduced in the paper.

```
Dialects Frequency Lists/
│
├── Bivalency Removed (dialect - MSA)/
│ allEGY_minusMSA.txt
│ allGLF_minusMSA.txt
│ allLAV_minusMSA.txt
│ allNOR_minusMSA.txt
│
├── Dialects’ MSA/
│ allEGY_dialectal_MSA.txt
│ allGLF_dialectal_MSA.txt
│ allLAV_dialectal_MSA.txt
│ allNOR_dialectal_MSA.txt
│
├── Dialects Tokens WITH Frequency Count/
│ all-EGY_FreqList.txt
│ all-GLF_FreqList.txt
│ all-LAV_FreqList.txt
│ all-MSA_FreqList.txt
│ all-NOR_FreqList.txt
│
└── Dialects Tokens NO Frequency Count/
all-EGY_FreqList2.txt
all-GLF_FreqList2.txt
all-LAV_FreqList2.txt
all-MSA_FreqList2.txt
all-NOR_FreqList2.txt
```

**These include:**

- **Bivalency-removed lists:** dialect-specific vocabularies after removing shared (bivalent) words  
- **Dialectal-MSA lists:** vocabulary shared with MSA via written code-switching  
- **Frequency lists:** token frequency counts  
- **Non-frequency lists:** raw token lists without counts  

---

## 📊 **CSV Conversion + Train/Dev/Test Splits**
The original text files were converted into a unified sentence-level CSV: arabic_dialects_full.csv


with the schema:

| sentence | dialect |
|----------|---------|
| … | EGY / GLF / LAV / MSA / NOR |

Each sentence corresponds to one line from the original files.

This CSV was then **stratified and split** into:
arabic_dialects_train.csv
arabic_dialects_dev.csv
arabic_dialects_test.csv



Splits preserve dialect balance and follow an 80/10/10 ratio.

These files power the `csv_splits` configuration for immediate model training.

---

## 🧪 Intended Use

This dataset supports several research tasks:

### **Dialect Identification**
- Train machine-learning models to classify EGY/GLF/LAV/NOR/MSA  
- Evaluate performance on highly bivalent and lexically overlapping dialects  
- Benchmark features beyond n-grams  

### **Bivalency & Code-Switching Analysis**
- Study how words shift between dialects and MSA  
- Explore written code-switching in online text and commentary discourse  

### **Linguistic Feature Engineering**
- Use SBP lists as interpretable features  
- Combine stylistic, grammatical, and frequency-based signals  

---

## 📊 Data Statistics

(From the original publication)

| Dialect | Sentences | Words |
|--------|-----------|-------|
| EGY | 4,061 | 118,152 |
| GLF | 2,546 | 65,752 |
| LAV | 2,463 | 67,976 |
| MSA | 3,731 | 49,985 |
| NOR | 3,693 | 53,204 |
| **Total** | **16,494** | **355,069** |

---

## 🔍 Example Usage

### Load the full text for dialect classification

```python
from datasets import load_dataset

ds = load_dataset("YOUR_REPO_NAME", "full_text")
print(ds["train"][0])
ds = load_dataset("YOUR_REPO_NAME", "freq_lists")
freq_list = ds["EGY_freq"]
```

---
## ⚠ Licence

This dataset is released for research purposes only.
Texts originate from publicly available online sources or earlier datasets where redistribution for research is permitted.

---

## 🙏 Acknowledgements

This dataset was developed at UCREL, Lancaster University, as part of research into Arabic dialect variation, bivalency, and automatic dialect identification.