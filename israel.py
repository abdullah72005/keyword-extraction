from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_documents(documents, top_n=10):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()


    # Get the first document's data
    first_doc_vector = tfidf_matrix[-1]

    # Convert to a dictionary for easy reading
    scores = {feature_names[i]: first_doc_vector[0, i] for i in first_doc_vector.indices}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_items = sorted_scores[:top_n]

    print(f"{'Keyword':<20} | {'Score':<10}")
    print("-" * 32)
    for word, score in top_items:
        print(f"{word:<20} | {score:.4f}")

    return top_items