# CS329 Project: Context-Aware Arabic-English Translation (or whatever)
cs 329 proj
(basic summary from slides)

## Problem Statement

  * Dialects like Maghrebi Arabic are underrepresented in training datasets, leading to inaccurate translations by LLMs.
  * Current LLM translations often use direct, literal translations, lack human nuance, and may use words that do not agree with the target language context.

## Project Overview

  * The project is an Arabic-to-English translation model that uses Modern Standard Arabic (MSA) as its baseline.
  * It extends accurate, natural translation from four dialects: Egyptian, Gulf, Levantine, and Maghrebi.
  * Dialect identification will be handled by integrating and validating an existing dialect classifier.

## Methodology

1.  **Training:** Use HuggingFace to train and cross-validate the model on verified translations.
2.  **Testing:** Choose testing text samples and refine prompts to perform translations.
3.  **Human Review:** Verify translation accuracy with native speakers.
4.  **Final Product:** Test the model against several kinds of text.

## Innovation and Impact

  * **Approach:** Dialect is detected first, so the translation reflects natural speech, not a formal MSA approximation.
  * **Significance:** Prevents changes in meaning, tone, and intent that result from dialect confusion.
  * **Scalability:** The approach is generalizable and replicable for other dialect-heavy languages, such as Spanish dialects.
  * **Accessibility:** Intended to be accessible as a browser extension.
  * **Beneficiaries:** Journalists, researchers, developers, and nearly 400 million Arabic speakers.

## Project Timeline (Key Milestones)

  * **3/18:** Finalize the research question, dataset choice, and evaluation plan.
  * **3/23:** Extract Arabic-English sentence pairs from Huggingface datasets and convert data to clean CSV training/testing splits.
  * **4/1:** Train the model, complete prompt engineering, generate first sample outputs, and compare outputs using native human review.
  * **4/7:** Finalize prompts through iterative feedback and start testing on an expanded range of texts.
  * **4/20:** Deliver the live demo and final project report with polished end-to-end examples.

## Team Roles

  * **Anushka Basu:** Initial model design and prompt engineering.
  * **Charles Cook:** Refinement of the model and evaluation design.
  * **Ethan Hsiung:** Programming aspect of the project and shared work on the language model.
  * **Sumaya Abdi:** Data sourcing and human verification.
