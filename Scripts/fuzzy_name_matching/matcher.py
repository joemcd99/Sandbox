import pandas as pd
from rapidfuzz import process, fuzz
import sys
import argparse

def fuzzy_match(df_a, df_b, col_a, col_b, threshold=80, output_file='matches.csv'):
    """
    Matches names from two DataFrames using fuzzy matching.
    """

    # Ensure columns exist
    if col_a not in df_a.columns:
        print(f"Error: Column '{col_a}' not found in DataFrame A")
        return
    if col_b not in df_b.columns:
        print(f"Error: Column '{col_b}' not found in DataFrame B")
        return

    # Convert to string and handle NaNs
    names_a = df_a[col_a].astype(str).fillna('')
    names_b = df_b[col_b].astype(str).fillna('')
    
    # Create a mapping of name -> index/row for lookup if needed, 
    # but for now we just want to find matches for A in B.
    
    results = []
    
    print(f"Matching {len(names_a)} names from DataFrame A against {len(names_b)} names from DataFrame B...")
    
    # Iterate through names in A and find best match in B
    for idx, name in names_a.items():
        if not name.strip():
            continue
            
        # Approach 1: Token Sort (Good for "Smith, John" vs "John Smith")
        match_sort = process.extractOne(
            name, names_b, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
        )
        
        # Approach 2: Token Set (Good for "John Smith" vs "John Smith (CEO)")
        match_set = process.extractOne(
            name, names_b, scorer=fuzz.token_set_ratio, score_cutoff=threshold
        )

        # Logic: Pick the method that gave the higher score
        best_match = None
        method_used = "None"

        if match_sort and match_set:
            if match_set[1] > match_sort[1]:
                best_match = match_set
                method_used = "token_set"
            else:
                best_match = match_sort
                method_used = "token_sort"
        elif match_sort:
            best_match = match_sort
            method_used = "token_sort"
        elif match_set:
            best_match = match_set
            method_used = "token_set"
        
        if best_match:
            matched_name, score, match_idx = best_match
            results.append({
                'Original Name (A)': name,
                'Matched Name (B)': matched_name,
                'Score': score,
                'Method': method_used,
                'Index A': idx,
                'Index B': match_idx
            })
            
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        print(f"\nFound {len(results_df)} matches with score >= {threshold}:")
        print(results_df.to_string(index=False))
        
        # Optional: Save to CSV
        results_df.to_csv(output_file, index=False)
        print(f"\nMatches saved to {output_file}")
    else:
        print("\nNo matches found above the threshold.")

if __name__ == "__main__":
    # Create inline DataFrames
    data_a = {
        'id': [1, 2, 3, 4, 5],
        'name': ['John Smith', 'Jane Doe', 'Robert Johnson', 'Michael Brown', 'Emily Davis']
    }
    df_a = pd.DataFrame(data_a)

    data_b = {
        'id': [101, 102, 103, 104, 105, 106],
        'full_name': ['Smith, John', 'J. Doe', 'Bob Johnson', 'Mike Brown', 'Emily J. Davis', 'Unmatched Person']
    }
    df_b = pd.DataFrame(data_b)

    parser = argparse.ArgumentParser(description="Fuzzy match names between two inline DataFrames.")
    parser.add_argument("--threshold", type=int, default=80, help="Matching threshold (0-100)")
    parser.add_argument("--output", default="matches.csv", help="Output CSV filename")
    
    # Handle running in Jupyter/IPython where sys.argv contains kernel args
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    # Pass the inline dataframes and specify the column names directly
    fuzzy_match(df_a, df_b, 'name', 'full_name', args.threshold, args.output)
