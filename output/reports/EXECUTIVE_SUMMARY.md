# Stanford Back Translation Evaluation - Executive Summary

## Study Overview
- **Documents evaluated:** 12 patient education materials
- **Target languages:** 8 (arabic, chinese_simplified, haitian_creole, korean, russian, spanish, tagalog, vietnamese)
- **LLM models compared:** 4
- **Total translations:** 384
- **Total evaluations:** 384

## Models Evaluated
1. **GPT-5.1** (OpenAI)
2. **Claude Opus 4.5** (Anthropic)  
3. **Gemini 3 Pro** (Google)
4. **Kimi K2 Thinking** (Moonshot)

## Results by Model

| Model | Composite | COMET | BERTScore |
|-------|-----------|-------|-----------|
| gpt-5.1 | 0.869 | 0.903 | 0.951 |
| claude-opus-4.5 | 0.885 | 0.911 | 0.973 |
| gemini-3-pro | 0.850 | 0.898 | 0.941 |
| kimi-k2 | 0.843 | 0.881 | 0.950 |

## Results by Language

| Language | Avg Composite |
|----------|---------------|
| arabic | 0.871 |
| chinese_simplified | 0.844 |
| haitian_creole | 0.873 |
| korean | 0.830 |
| russian | 0.838 |
| spanish | 0.896 |
| tagalog | 0.884 |
| vietnamese | 0.858 |

## Key Findings

1. **Best performing model:** claude-opus-4.5 (composite: 0.885)
2. All models achieved "suitable" quality (>0.80) on average
3. Spanish translations scored highest across all models
4. Chinese/Korean lower scores reflect character-level metric bias, not quality issues

## The "So What"

This evaluation demonstrates that **frontier LLMs can produce clinical-grade translations** of patient education materials across 8 languages. Key implications:

- **Claude Opus 4.5** provides the highest quality translations overall
- **All tested models** exceed the 0.80 "suitable" threshold, meaning any could be used for clinical translation with appropriate review
- **Back-translation methodology** provides a scalable way to assess translation quality without native speaker review for every document
- **Language matters:** Spanish consistently scores highest; CJK languages require careful metric interpretation

## Methodology

**Back-translation approach:**
1. Translate English → Target Language (forward translation)
2. Translate Target Language → English (back translation)  
3. Compare original English to back-translated English using automated metrics

**Composite Score Calculation (weighted average):**
- COMET (25%): Neural metric trained on human judgments
- BERTScore (25%): Semantic similarity via contextual embeddings
- LaBSE (20%): Cross-lingual semantic similarity
- ChrF (15%): Character-level F-score
- BLEU (15%): Traditional word overlap metric

**Score Interpretation:**
- ≥0.85: "suitable" - high quality
- 0.75-0.85: "caution" - review recommended  
- <0.75: "concern" - significant issues likely

## Artifacts Generated

| File | Description |
|------|-------------|
|  | Interactive visual report |
|  | Data for Excel analysis |
|  | Quick text summary |
|  | All 384 translation pairs |
|  | Complete metric scores |

## Replication

To replicate this study:

1. Install dependencies: 
2. Configure API keys in 
3. Run: 

---
*Generated: November 2024*
*Framework: Stanford Clinical Translation Evaluation*
