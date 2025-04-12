# preprocessing.py

import re
import spacy

# Load spaCy Portuguese model once
nlp = spacy.load("pt_core_news_sm")

# Dictionary for acronym expansion
acronyms = {
    "Selic": "Sistema Especial de Liquidação e de Custódia",
    "PIB": "Produto Interno Bruto",
    "CDI": "Certificado de Depósito Interbancário",
    "LPRs": "Loan Prime Rates",
    "Ibovespa": "Índice Bovespa",
    "BB": "Banco do Brasil",
    "BC": "Banco Central",
    "FGTS": "Fundo de Garantia do Tempo de Serviço",
    "STF": "Supremo Tribunal Federal",
    "CPI": "Índice de Preços ao Consumidor",
    "MP": "Medida Provisória",
    "EUA": "Estados Unidos",
    "ONU": "Organização das Nações Unidas",
    "FGV": "Fundação Getúlio Vargas",
    "IBGE": "Instituto Brasileiro de Geografia e Estatística",
    "BNDES": "Banco Nacional de Desenvolvimento Econômico e Social",
    "IPCA": "Índice Nacional de Preços ao Consumidor Amplo",
    "DI": "Depósito Interfinanceiro",
    "IR": "Imposto de Renda",
    "OI": "Operadora Oi",
    "CV": "Câmara de Vereadores"
}

# Noisy acronyms to remove
noisy_acronyms = {"ROMI", "ENEVA", "LIGHT", "DA"}

def normalize_numbers(text):
    text = re.sub(r"R\$ ?([\d.,]+) bilhões", r"\1B", text)
    text = re.sub(r"R\$ ?([\d.,]+) milhões", r"\1M", text)
    text = re.sub(r"([\d.,]+) pp", r"\1%", text)
    text = text.replace(",", "")
    return text

def expand_acronyms(text, acronym_dict):
    for acronym, full_form in acronym_dict.items():
        text = re.sub(rf'\b{re.escape(acronym)}\b', full_form, text, flags=re.IGNORECASE)
    return text

def remove_noisy_acronyms(text, noisy_set):
    return re.sub(r'\b(?:' + '|'.join(noisy_set) + r')\b', '', text)

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def preprocess_text(text):
    text = normalize_numbers(text)
    text = expand_acronyms(text, acronyms)
    text = remove_noisy_acronyms(text, noisy_acronyms)
    text = lemmatize_text(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()