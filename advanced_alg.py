from keybert import KeyBERT

# 1. Initialize the model 
# 'all-MiniLM-L6-v2' is a lightweight BERT model that is fast 
# and very accurate for keyword extraction.
kw_model = KeyBERT(model='all-MiniLM-L6-v2')


def extract_keywords(text, top_n=10):
    # 2. Extract keywords
    # We use the model to extract keywords from the document.
    # top_n specifies how many keywords we want to extract.
    keywords = kw_model.extract_keywords(
        text,
        stop_words='english',
        top_n=top_n,
    )
    
    # Human-readable printing
    print(f"{'Keyword':<20} | {'Score':<10}")
    print("-" * 32)
    for word, score in keywords:
        print(f"{word:<20} | {score:.4f}")

    return keywords