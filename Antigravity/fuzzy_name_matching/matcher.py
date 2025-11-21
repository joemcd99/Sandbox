import pandas as pd
from rapidfuzz import process, fuzz
import sys

def fuzzy_match(file_a, file_b, col_a, col_b, threshold=80):
    """
    Matches names from two CSV files using fuzzy matching.
    """
    try:
        df_a = pd.read_csv(file_a)
        df_b = pd.read_csv(file_b)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Ensure columns exist
    if col_a not in df_a.columns:
        print(f"Error: Column '{col_a}' not found in {file_a}")
        return
    if col_b not in df_b.columns:
        print(f"Error: Column '{col_b}' not found in {file_b}")
        return

    # Convert to string and handle NaNs
    names_a = df_a[col_a].astype(str).fillna('')
    names_b = df_b[col_b].astype(str).fillna('')
    
    # Create a mapping of name -> index/row for lookup if needed, 
    # but for now we just want to find matches for A in B.
    
    results = []
    
    print(f"Matching {len(names_a)} names from '{file_a}' against {len(names_b)} names from '{file_b}'...")
    
    # Iterate through names in A and find best match in B
    for idx, name in names_a.items():
        if not name.strip():
            continue
            
        # extractOne returns (match, score, index)
        match = process.extractOne(
            name, 
            names_b, 
            scorer=fuzz.token_sort_ratio, # Good for "Smith, John" vs "John Smith"
            score_cutoff=threshold
        )
        
        if match:
            matched_name, score, match_idx = match
            results.append({
                'Original Name (A)': name,
                'Matched Name (B)': matched_name,
                'Score': score,
                'Index A': idx,
                'Index B': match_idx
            })
            
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        print(f"\nFound {len(results_df)} matches with score >= {threshold}:")
        print(results_df.to_string(index=False))
        
        # Optional: Save to CSV
        output_file = 'matches.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nMatches saved to {output_file}")
    else:
        print("\nNo matches found above the threshold.")

if __name__ == "__main__":
    # For demonstration, using the dummy files we created
    # In a real scenario, these could be command line arguments
    FILE_A = 'data_source_a.csv'
    FILE_B = 'data_source_b.csv'
    COL_A = 'name'
    COL_B = 'full_name'
    
    fuzzy_match(FILE_A, FILE_B, COL_A, COL_B)
