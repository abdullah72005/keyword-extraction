import pandas as pd
from preprocessing import preprocess_text
from israel import vectorize_documents
from load_dataset import load_dataset
from test_movie_names_with_indices import movies
from advanced_alg import extract_keywords

def main():
    df = load_dataset()

    while True:
        alg = input("""\n\n\n\n\n\n\nWhich algorithm would you like to use for keyword extraction?


1. TF-IDF
2. KeyBERT (Advanced) 
3. Both
Please enter the number corresponding to your choice: """)
        if alg in ['1', '2', '3']:
            break

    print(f"\n\n\n\n{movies}\n\n\n\n")

    while True:
        movie_index = input("Please enter the index of the document you want to analyze: ")
        if movie_index.isdigit() and 0 <= int(movie_index) < 200:
            break
        else:
            print("Invalid input. Please enter a valid index.")

    movie = pd.read_json("datasett/test.jsonl", lines=True)
    movie = preprocess_text(movie["summary"][int(movie_index)])
    movie = pd.Series(movie)
    df = pd.concat([df, movie], keys=['original', 'new'])
    
    if alg == '1':
        print("\n\n\n\n\n\n\n\n\n")
        vectorize_documents(df)
    elif alg == '2':
        print("\n\n\n\n\n\n\n\n\n")
        extract_keywords(movie[0])
    elif alg == '3':
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nTF-IDF Results:")
        vectorize_documents(df)
        print("\n\n\n\nKeyBERT Results:")
        extract_keywords(movie[0])
        print("\n")


if __name__ == "__main__":    
    main()