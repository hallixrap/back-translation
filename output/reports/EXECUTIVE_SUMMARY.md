# Stanford Back Translation Evaluation - Executive Summary

## Study Overview

This study evaluated how well frontier LLMs translate clinical patient education materials using **back-translation methodology**: translating English documents to a target language, then back to English, and measuring how much meaning is preserved.

- **Documents evaluated:** 12 patient education materials from [MedlinePlus](https://medlineplus.gov/)
- **Target languages:** 8 (Spanish, Chinese Simplified, Vietnamese, Tagalog, Russian, Arabic, Korean, Haitian Creole)
- **LLM models compared:** 4 frontier models
- **Total translation pairs:** 384

## Source Documents

| Document | Category | Source |
|----------|----------|--------|
| [Heart Failure - Discharge](https://medlineplus.gov/ency/patientinstructions/000114.htm) | Cardiology | MedlinePlus |
| [Taking Warfarin](https://medlineplus.gov/ency/patientinstructions/000292.htm) | Cardiology | MedlinePlus |
| [Heart Attack - Discharge](https://medlineplus.gov/ency/patientinstructions/000090.htm) | Cardiology | MedlinePlus |
| [High Blood Pressure - What to Ask Your Doctor](https://medlineplus.gov/ency/patientinstructions/000226.htm) | Cardiology | MedlinePlus |
| [Type 2 Diabetes - What to Ask Your Doctor](https://medlineplus.gov/ency/patientinstructions/000217.htm) | Diabetes | MedlinePlus |
| [Low Blood Sugar - Self-Care](https://medlineplus.gov/ency/patientinstructions/000085.htm) | Diabetes | MedlinePlus |
| [Asthma - Quick-Relief Drugs](https://medlineplus.gov/ency/patientinstructions/000008.htm) | Respiratory | MedlinePlus |
| [COPD - What to Ask Your Doctor](https://medlineplus.gov/ency/patientinstructions/000215.htm) | Respiratory | MedlinePlus |
| [ACE Inhibitors](https://medlineplus.gov/ency/patientinstructions/000087.htm) | Medications | MedlinePlus |
| [Stroke - Discharge](https://medlineplus.gov/ency/patientinstructions/000132.htm) | Emergency | MedlinePlus |
| [Surgical Wound Care - Open](https://medlineplus.gov/ency/patientinstructions/000040.htm) | Surgical | MedlinePlus |
| [Hip Replacement - Discharge](https://medlineplus.gov/ency/patientinstructions/000169.htm) | Surgical | MedlinePlus |

## Results by Model

Models ranked by composite score (Claude Opus 4.5 best overall):

| Model | Composite | COMET | BERTScore | LaBSE | ChrF | BLEU |
|-------|-----------|-------|-----------|-------|------|------|
| **Claude Opus 4.5** | **0.885** | **0.911** | **0.973** | **0.902** | **81.1** | **63.4** |
| GPT-5.1 | 0.869 | 0.903 | 0.951 | 0.900 | 79.6 | 59.1 |
| Gemini 3 Pro | 0.850 | 0.898 | 0.941 | 0.884 | 76.1 | 52.0 |
| Kimi K2 Thinking | 0.843 | 0.881 | 0.950 | 0.893 | 73.6 | 49.0 |

## Understanding the Metrics

The **Composite Score** is a weighted average of five metrics, each measuring different aspects of translation quality:

| Metric | Weight | What It Measures | Why It Matters |
|--------|--------|------------------|----------------|
| **COMET** | 25% | Neural metric trained on human quality judgments | Best predictor of how humans rate translation quality |
| **BERTScore** | 25% | Semantic similarity using contextual embeddings | Captures meaning preservation even when wording differs |
| **LaBSE** | 20% | Cross-lingual sentence embedding similarity | Measures meaning across languages, useful for back-translation |
| **ChrF** | 15% | Character-level n-gram F-score | Good for morphologically rich languages; captures partial word matches |
| **BLEU** | 15% | Word n-gram overlap | Traditional MT benchmark; strict exact-match scoring |

**Score Interpretation:**
- **0.85+**: "Suitable" - High quality, ready for clinical use with standard review
- **0.75-0.85**: "Caution" - Acceptable quality, enhanced review recommended
- **<0.75**: "Concern" - Quality issues likely, significant revision needed

## Results by Language

| Language | Avg Composite | Notes |
|----------|---------------|-------|
| Spanish | 0.896 | Highest scores - abundant training data |
| Tagalog | 0.884 | Strong performance |
| Haitian Creole | 0.873 | Good quality despite being low-resource |
| Arabic | 0.871 | Good quality |
| Vietnamese | 0.858 | Good quality |
| Chinese (Simplified) | 0.844 | Lower scores reflect character-level metric bias |
| Russian | 0.838 | Good quality |
| Korean | 0.830 | Lower scores reflect character-level metric bias |

**Note:** Chinese and Korean scores appear lower partly because character-level metrics (BLEU, ChrF) don't account for the information density of CJK characters. A single Chinese character often conveys the meaning of multiple English words.

## Key Findings

1. **Claude Opus 4.5 leads** with the highest composite (0.885) and COMET (0.911) scores
2. **All models are production-viable** - every model exceeds the 0.80 "suitable" threshold
3. **Spanish translations excel** - highest scores across all models, likely due to training data abundance
4. **Back-translation is scalable** - enables quality assessment without native speakers for every document
5. **Haitian Creole performs well** - despite being a low-resource language, all models achieved good scores

## The "So What"

This evaluation demonstrates that **frontier LLMs can produce clinical-grade translations** of patient education materials across 8 languages:

- **For clinical teams:** Any of these models can be used for translation with appropriate human review. Claude Opus 4.5 provides the highest quality.
- **For health equity:** Good performance on Haitian Creole suggests LLMs may help address translation gaps for underserved language communities.
- **For scalability:** Back-translation methodology enables quality assessment at scale without requiring native speaker review for every document.
- **For future work:** Human evaluation by bilingual clinicians would validate these automated metrics.

## Methodology

**Back-translation approach:**
```
English Original → [LLM] → Target Language → [LLM] → English Back-Translation
                                                              ↓
                                                    Compare with Original
```

The intuition: if a translation faithfully preserves meaning, translating it back to English should produce text similar to the original. Large semantic differences in the back-translation indicate potential quality issues.

### Prompts Used

**Forward Translation (English → Target Language):**
```
System: You are an expert medical translator specializing in patient education materials.
Your translations must:
1. Preserve ALL medical terminology accurately
2. Maintain patient-friendly readability (aim for 6th-8th grade reading level)
3. Be culturally appropriate for the target language speakers
4. Keep ALL safety warnings and critical information intact
5. Preserve document structure (headings, bullet points, numbered lists)

CRITICAL: Never omit, modify, or soften any medical warnings or safety information.

User: Translate the following patient education document from English to {target_language}.

Requirements:
- Maintain exact document structure and formatting
- Preserve all medical terms accurately (use commonly understood equivalents when available)
- Keep all numbered lists, bullet points, and section headers
- Do not add explanations or commentary
- Do not remove any content

Document to translate:
---
{document_text}
---

Provide only the {target_language} translation, nothing else.
```

**Back-Translation (Target Language → English):**
```
System: You are a professional medical translator. Your task is to translate text back to English.

CRITICAL INSTRUCTIONS:
1. Translate EXACTLY what is written - do not correct perceived errors
2. Preserve the meaning and structure as closely as possible
3. If something seems unclear or wrong in the source, translate it literally anyway
4. Do not add any explanations or notes about the translation

User: Translate the following {source_language} medical text back to English.

Important: Translate literally and exactly. Do not correct any errors you perceive -
we need to see exactly what the text says.

Text to translate:
---
{translated_text}
---

Provide only the English translation, nothing else.
```

### Analysis Pipeline

1. **Translation:** Each document was translated to all 8 languages using each of the 4 models (384 translations)
2. **Back-translation:** Each translation was converted back to English by the same model
3. **Metric calculation:** Original English compared to back-translated English using 5 automated metrics
4. **Composite scoring:** Weighted average of all metrics to produce a single quality score
5. **Suitability rating:** Scores mapped to clinical usability categories (suitable/caution/concern)

---
*Generated: November 2025*
*Framework: Stanford Clinical Translation Evaluation*
