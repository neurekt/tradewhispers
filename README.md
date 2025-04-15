### **USD/BRL News Classification and Trading Signal Generator**
This project applies natural language processing (NLP) techniques to extract market-moving insights from Brazilian Portuguese financial news, with the objective of generating actionable trading signals for the USD/BRL currency pair. The classifier was developed as part of a collaborative research initiative with NYU Courant Institute of Mathematical Sciences.

**Classification**
Each news article is labeled based on its likely directional impact on the currency pair, from the perspective of a trader:
- +1: Favorable to the BRL
- -1: Favorable to the USD
- 0: Neutral / Favorable to Neither

**Macroeconomic Framework:**
USD-positive signals typically stem from:
- Monetary tightening by the Federal Reserve, which attracts global capital inflows via higher real yields.
- Robust economic data such as rising GDP, strong job creation, low inflation, and high demand for U.S. Treasuries.
- Global risk aversion, where capital seeks safe-haven assets, leading to USD strength.

BRL-positive signals are driven by:
- High domestic interest rates, incentivizing foreign investment in Brazilian assets.
- Commodity booms (e.g., soybeans, iron ore, coffee), improving Brazil’s trade balance.
- Infrastructure investment, capital inflows, and signs of political and fiscal stability.

**Why PT-BR Language Processing Matters**
- Preserves financial nuance and regional context critical to interpreting sentiment correctly.
- Avoids distortions introduced by machine translation.
- Improves domain accuracy by leveraging specialized, Portuguese-language financial vocabulary.

**Project Background**
I engineered an end-to-end classification pipeline using a dataset sourced from BDM, Valor Econômico, and the Bloomberg Terminal. I experimented with three strategies:
- Averaged Word2Vec embeddings
- Custom word embeddings (pre-trained and fine-tuned)
- BERT embeddings

**Challenges Faced**
- Striking a balance between nuance and model simplicity: Since we kept the articles in Portuguese, we ran into challenges with training stability and explainability — especially when trying to map subtle macroeconomic cues to discrete labels.
- Subjective labeling: Some headlines were tough to classify, either because they touched on multiple conflicting signals or were written in a vague tone. This made consistency a recurring issue during model evaluation.

**What I’d Do Differently**
- I’d design a lightweight interface for analysts or traders to browse the model’s classifications and drill into the reasoning behind each one. This would make it easier to spot misclassifications and add a feedback loop.
- Instead of relying solely on single-label classification, I’d allow the model to assign multiple labels or at least express uncertainty — for example, when an article mentions both U.S. interest rates and Brazilian commodities.
- We're now exploring prompt engineering with open-source LLaMA models, which we think will better capture the type of logic a human economist would use. We're batching examples during inference to give the model more context and reduce variance.
- I also plan to experiment with a rule-based lexicon approach, which we haven’t implemented yet. This could serve as a complementary baseline, especially for simpler headlines where sentiment tends to be clearer.

While results were directionally promising, model precision was limited by linguistic ambiguity and the small labeled dataset. We are now transitioning to a zero-shot LLM classification approach (using open-source LLaMA models) to better emulate the nuanced reasoning of our economist counterpart in Brazil.

