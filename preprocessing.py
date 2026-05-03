import spacy 

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)

    # 1. Only merge USEFUL entity types (Ignore DATE, MONEY, PERCENT, TIME)
    valid_ent_types = {"ORG", "PERSON", "PRODUCT", "GPE", "LOC", "WORK_OF_ART"}

    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            if ent.label_ in valid_ent_types:
                retokenizer.merge(ent)

    tokens = []
    for token in doc:
        if (not token.is_stop and 
            not token.is_punct and 
            not token.is_space and 
            not token.like_num and 
            not token.is_currency and
            token.text != "%"): 
            
            if token.ent_type_ in valid_ent_types:
                entity_text = token.text.lower().replace(" ", "")
                tokens.append(entity_text)
            else:
                tokens.append(token.lemma_.lower())

    
    
    return " ".join(tokens)

