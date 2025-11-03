# 🚀 Smart Resume Finder 🚀

This is a smart resume-matching tool built with Python, SentenceTransformers (SBERT), and Gradio.

It allows a user to upload a CSV of resumes and find the best matches for a given job description using semantic similarity (TF-IDF or SBERT).

## 🎈 Live Demo

**You can try the app live on Hugging Face Spaces:**
**[https://huggingface.co/spaces/Vinaykoushik/resume-finder](https://huggingface.co/spaces/Vinaykoushik/resume-finder)**



## ✨ Features
* **Live Deployment:** Hosted and permanently available on Hugging Face Spaces.
* **Dual Models:** Supports both fast **TF-IDF** and accurate **SBERT** (`all-MiniLM-L6-v2`) models.
* **On-the-Fly Indexing:** Builds the vector indexes in real-time when a user uploads their CSV.
* **Interactive UI:** Built with Gradio, allowing file uploads, text input, and interactive plots.
* **Metrics:** Can compute Precision, Recall, and F1-score for the top K results.

## 🔧 How to Run Locally

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the app:
    ```bash
    python app.py
    ```

## 📁 Files in this Repository

* `app.py`: The main Python script containing all the logic and the Gradio app.
* `requirements.txt`: The list of all Python libraries required to run the app.
