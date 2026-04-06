import os
import glob
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import plotly.express as px
import re
import textwrap

# 1. Configuration
DIRECTORY_PATH = './'  # Replace with your directory path
COLUMN_NAME = 'full_response'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
HIGHLIGHT_FILENAME = 'endgame_deception_results_9b_baseline'

def main():
    # 2. Load all CSVs from the directory
    csv_files = glob.glob(os.path.join(DIRECTORY_PATH, '*.csv'))
    
    if not csv_files:
        print("No CSV files found in the specified directory.")
        return

    print(f"Found {len(csv_files)} CSV files. Loading data...")
    
    df_list = []
    for file in csv_files:
        try:
            temp_df = pd.read_csv(file, usecols=[COLUMN_NAME])
            temp_df['source_file'] = os.path.basename(file)
            df_list.append(temp_df)
        except ValueError:
            print(f"Skipping {file}: column '{COLUMN_NAME}' not found.")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not df_list:
        print("No valid data loaded. Exiting.")
        return

    # Combine all dataframes into one
    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=[COLUMN_NAME])
    df[COLUMN_NAME] = df[COLUMN_NAME].astype(str)
    
    # Extract text between <think> and </think>
    print("Extracting text between <think> tags...")
    df['extracted_thought'] = df[COLUMN_NAME].str.extract(r'(?s)(.*?)</think>', expand=False)
    
    # Clean up
    df = df.dropna(subset=['extracted_thought'])
    df['extracted_thought'] = df['extracted_thought'].str.strip()
    df['is_baseline'] = df['source_file'].str.contains(HIGHLIGHT_FILENAME, na=False)
    
    # --- NEW: Format data for Plotly ---
    # Create a clean 'Category' column for the legend
    df['Category'] = 'Other Files'
    df.loc[df['is_baseline'], 'Category'] = 'Baseline (9b_battleship)'
    
    # Create a wrapped hover text column (Plotly uses HTML <br> for line breaks)
    # We also truncate at 500 characters so massive thoughts don't break the UI
    def format_hover_text(text):
        truncated = text[:500] + ('...' if len(text) > 500 else '')
        wrapped = textwrap.wrap(truncated, width=80)
        return '<br>'.join(wrapped)
        
    df['hover_text'] = df['extracted_thought'].apply(format_hover_text)
    
    responses = df['extracted_thought'].tolist()
    print(f"Total valid <think> blocks to embed: {len(responses)}")

    if len(responses) == 0:
        print("No <think> tags were found in the data. Exiting.")
        return

    # 3. Generate Embeddings
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Generating embeddings (this might take a moment)...")
    embeddings = model.encode(responses, show_progress_bar=True)

    # 4. Dimensionality Reduction (t-SNE)
    print("Reducing dimensions from high-D to 2D using t-SNE...")
    perplexity_value = min(30, len(embeddings) - 1) 
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    df['tsne_x'] = embeddings_2d[:, 0]
    df['tsne_y'] = embeddings_2d[:, 1]

    # 5. Plot the Interactive Dots using Plotly
    print("Generating interactive plot...")
    
    fig = px.scatter(
        df,
        x='tsne_x',
        y='tsne_y',
        color='Category',
        color_discrete_map={
            'Baseline (9b_battleship)': '#d62728', # Red
            'Other Files': '#1f77b4'               # Blue
        },
        hover_data={'tsne_x': False, 'tsne_y': False, 'Category': False, 'hover_text': True},
        title='Interactive 2D Visualization of "<think>" Block Embeddings'
    )
    
    # Tell Plotly to display the 'hover_text' column when hovering
    fig.update_traces(hovertemplate="<b>Thought snippet:</b><br>%{customdata[0]}")
    
    # Save as an interactive HTML file
    fig.write_html('interactive_embeddings.html')
    print("Interactive plot saved as 'interactive_embeddings.html'. Open this file in your web browser!")
    
    # Automatically open the plot in your default browser
    fig.show()

if __name__ == "__main__":
    main()