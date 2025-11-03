import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# --- Core ML/NLP Libraries ---
# (These are assumed from your previous cells)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer

# --- 1. Load SBERT Model ---
# (This model must be defined to be used)
try:
    # Use 'all-MiniLM-L6-v2' as the default pretrained model
    sbert = SentenceTransformer('all-MiniLM-L6-v2') 
    print("Pretrained SBERT model loaded.")
except Exception as e:
    print(f"Error loading SBERT model: {e}")
    # Create a dummy object if loading fails, so app can still load
    class DummySBERT:
        def encode(self, *args, **kwargs):
            print("ERROR: SBERT model not loaded.")
            return np.random.rand(kwargs.get('batch_size', 1), 384)
    sbert = DummySBERT()

# --- 2. Define Helper Plot Function ---
# (This function was called but not defined in your original cell)
def show_bar(labels, values, title="Top K Matches"):
    """Creates a horizontal bar plot and returns the figure."""
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, align='center', color='#2b8cbe')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(reversed(labels)) # Invert to show top match at top
    ax.invert_yaxis() 
    ax.set_xlabel('Similarity Score')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, v in enumerate(reversed(values)):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
    plt.tight_layout()
    return fig

# --- 3. Refactored Retrieval Functions (Fine-tuned removed) ---

def build_indexes_from_uploaded(df_uploaded):
    """Builds and returns TF-IDF and SBERT indexes for the uploaded data."""
    
    # 1) TF-IDF
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
    X = vect.fit_transform(df_uploaded['Resume_str'].astype(str).tolist())
    knn_t = NearestNeighbors(n_neighbors=min(50, len(df_uploaded)), metric='cosine', algorithm='brute')
    knn_t.fit(X)

    # 2) Pretrained SBERT
    emb_pre = sbert.encode(df_uploaded['Resume_str'].tolist(), convert_to_numpy=True, show_progress_bar=True)
    knn_pre = NearestNeighbors(n_neighbors=min(50, len(df_uploaded)), metric='cosine', algorithm='brute')
    knn_pre.fit(emb_pre)

    return vect, knn_t, knn_pre

def run_retrieval_for_uploaded(df_uploaded, model_choice, jd, k, vect, knn_t, knn_pre):
    """Runs retrieval using the pre-built indexes."""
    
    if model_choice == 'TF-IDF + KNN':
        v = vect.transform([jd])
        distances, indices = knn_t.kneighbors(v, n_neighbors=min(k, len(df_uploaded)))
    
    elif model_choice == 'SBERT (pretrained) + KNN':
        vec = sbert.encode([jd], convert_to_numpy=True, show_progress_bar=False)
        distances, indices = knn_pre.kneighbors(vec, n_neighbors=min(k, len(df_uploaded)))
    
    else:
        # This case should no longer be reachable
        raise ValueError("Invalid model choice.")

    sims = 1 - distances[0]
    results = df_uploaded.iloc[indices[0]].copy().reset_index().rename(columns={'index':'resume_index'})
    results['Similarity'] = sims

    # --- Prepare display ---
    display_cols = []
    if 'ID' in results.columns: display_cols.append('ID')
    if 'Name' in results.columns: display_cols.append('Name')
    if 'Category' in results.columns: display_cols.append('Category')
    display_cols.append('Similarity')
    
    # Ensure all display_cols exist, add if missing
    for col in display_cols:
        if col not in results.columns:
            results[col] = 'N/A'
            
    display = results[display_cols].head(k)

    # --- Build plot ---
    labels = results['Name'].tolist() if 'Name' in results.columns else results['resume_index'].astype(str).tolist()
    fig = show_bar(labels[:k], results['Similarity'].tolist()[:k], title=f"Top {k} matches ({model_choice})")

    # --- Prepare CSV download bytes ---
    csv_bytes = display.to_csv(index=False).encode()
    
    return display, fig, csv_bytes

# --- 4. Main Gradio App Function ---

def run_app(uploaded_file, model_choice, jd, k, compute_metrics, true_label):
    """
    Main function triggered by the 'Run' button.
    Reads the file, builds indexes, runs retrieval, and computes metrics.
    """
    if uploaded_file is None:
        return pd.DataFrame(), None, None, "🚫 Error: Please upload a CSV file first."
    if not jd or not jd.strip():
        return pd.DataFrame(), None, None, "🚫 Error: Please enter a Job Description."
        
    # --- Read and normalize CSV ---
    try:
        df_u = pd.read_csv(uploaded_file.name, engine='python')
    except Exception as e:
        return pd.DataFrame(), None, None, f"🚫 Error reading CSV: {e}"

    # Normalize columns
    cols = [c.strip() for c in df_u.columns]
    col_map = {}
    
    # Find Resume column
    resume_col = next((c for c in cols if any(tok in c.lower() for tok in ['resume','cv','text','profile','description','content','summary'])), None)
    if resume_col is None:
        return pd.DataFrame(), None, None, "🚫 Error: Couldn't find a resume column (e.g., 'Resume_str', 'text')."
    col_map[resume_col] = 'Resume_str'
    
    # Find optional columns
    if 'Name' not in cols:
        col_map[next((c for c in cols if 'name' in c.lower()), 'Name')] = 'Name'
    if 'ID' not in cols:
        col_map[next((c for c in cols if 'id' in c.lower()), 'ID')] = 'ID'
    if 'Category' not in cols:
        col_map[next((c for c in cols if any(tok in c.lower() for tok in ['category','role','position','dept'])), 'Category')] = 'Category'

    df_u = df_u.rename(columns=col_map)
    
    # Add placeholders if still missing
    if 'Name' not in df_u.columns: df_u['Name'] = "(Not Provided)"
    if 'ID' not in df_u.columns: df_u['ID'] = df_u.index
    if 'Category' not in df_u.columns: df_u['Category'] = "Unknown"

    df_u = df_u.dropna(subset=['Resume_str']).reset_index(drop=True)
    if df_u.empty:
        return pd.DataFrame(), None, None, "🚫 Error: No valid resume data found after loading."

    # --- Build indexes (This is the slow part) ---
    try:
        vect, knn_t, knn_pre = build_indexes_from_uploaded(df_u)
    except Exception as e:
        return pd.DataFrame(), None, None, f"🚫 Error building indexes: {e}"

    # --- Run retrieval ---
    display, fig, csv_bytes = run_retrieval_for_uploaded(df_u, model_choice, jd, k, vect, knn_t, knn_pre)

    # --- Optionally compute metrics ---
    metrics_text = "Metrics not computed (checkbox not selected or 'Category' column missing)."
    if compute_metrics:
        # Determine true label
        if true_label and str(true_label).strip():
            tlabel = str(true_label).strip()
        else:
            # Simple auto-detect (can be improved)
            all_cats = [str(c) for c in df_u['Category'].unique()]
            jd_low = jd.lower()
            tlabel = next((c for c in all_cats if c.lower() in jd_low), "Unknown")
            if tlabel == "Unknown":
                tlabel = df_u['Category'].mode().iloc[0]

        y_true = [tlabel] * len(display)
        y_pred = display['Category'].astype(str).tolist()
        
        p = precision_score(y_true, y_pred, average='micro', zero_division=0)
        r = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f = f1_score(y_true, y_pred, average='micro', zero_division=0)
        cls_report = classification_report(y_true, y_pred, zero_division=0, labels=np.unique(y_true + y_pred))
        
        metrics_text = (f"📈 Metrics (Top-K Results vs. True Label)\n"
                        f"----------------------------------------\n"
                        f"True Label Used: {tlabel}\n"
                        f"Micro-Precision: {p:.3f}\n"
                        f"Micro-Recall:    {r:.3f}\n"
                        f"Micro-F1-Score:  {f:.3f}\n\n"
                        f"Classification Report (Top-K):\n{cls_report}")

    # --- Prepare CSV for download ---
    tmp = None
    if csv_bytes is not None:
        tmp = BytesIO(csv_bytes)
        # This name is important for the download
        tmp.name = "top_k_results.csv" 

    return display, fig, tmp, metrics_text
# --- 5. Gradio UI (Aesthetic & Fine-tuned removed) ---
# (The functions like run_app, build_indexes_from_uploaded, etc. are all correct)
# (Paste this code to replace your existing gr.Blocks section)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange")) as demo_app:
    gr.Markdown(
        """
        # 🚀 Smart Resume Finder 🚀
        Upload your resume CSV, paste a job description, and find the best matches!
        
        **Instructions:**
        1.  Upload a CSV file containing your resumes. Must have a text column (e.g., 'Resume_str') and ideally 'Name' and 'Category'.
        2.  Select the matching model (TF-IDF for speed, SBERT for better accuracy).
        3.  Paste the job description.
        4.  Click 'Find Matches'.
        """
    )
    
    # --- FIX 1: Add an invisible State component to reliably hold the file data ---
    download_state = gr.State(value=None)
    
    # Use tabs for a clean layout
    with gr.Tabs():
        
        # --- TAB 1: Setup & Run ---
        with gr.TabItem("1. Setup & Run"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload = gr.File(label="Upload Resume CSV", file_types=[".csv"])
                    
                    model_choice = gr.Radio(
                        choices=['TF-IDF + KNN', 'SBERT (pretrained) + KNN'],
                        value='SBERT (pretrained) + KNN', 
                        label="🤖 Choose Retrieval Model"
                    )
                
                with gr.Column(scale=2):
                    jd_txt = gr.Textbox(lines=10, label="📋 Paste Job Description Here")

            with gr.Accordion("⚙️ Advanced Options (Metrics & Top-K)", open=False):
                k_slider = gr.Slider(1, 20, value=5, step=1, label="K (Number of results)")
                compute_metrics = gr.Checkbox(label="Compute Metrics (requires 'Category' column)", value=False)
                true_label_text = gr.Textbox(
                    label="True Label (Optional: for metrics)", 
                    placeholder="e.g., HR (if blank, will try to auto-detect)"
                )
            
            run_btn = gr.Button("🚀 Find Matches", variant="primary")

        # --- TAB 2: Results ---
        with gr.TabItem("2. View Results"):
            with gr.Row():
                result_table = gr.Dataframe(
                    headers=['ID', 'Name', 'Category', 'Similarity'], 
                    label="🏆 Top Matches"
                )
                result_plot = gr.Plot(label="📊 Similarity Plot")
            
            info_out = gr.Textbox(label="ℹ️ Info / Metrics Log", lines=10, interactive=False)
            
            download_btn = gr.Button("💾 Download Top-K CSV")

    # --- Button Click Logic ---
    
    # 1. Main run button
    run_btn.click(
        fn=run_app, 
        inputs=[upload, model_choice, jd_txt, k_slider, compute_metrics, true_label_text],
        
        # --- FIX 2: Output the file object (tmp) to download_state, NOT download_btn ---
        outputs=[result_table, result_plot, download_state, info_out]
    )

    # 2. Download button logic
    def download_results(tmpfile_from_state):
        if tmpfile_from_state is None:
            print("No file to download.")
            # You can raise an error to show the user
            gr.Warning("No file to download! Click 'Find Matches' first.")
            return None
        return tmpfile_from_state

    download_btn.click(
        fn=download_results, 
        # --- FIX 3: Get the file data from download_state, NOT download_btn ---
        inputs=[download_state], 
        outputs=[gr.File(label="Download CSV")]
    )

# --- Launch the App ---
# (share=True prints the shareable public URL)
demo_app.launch()
