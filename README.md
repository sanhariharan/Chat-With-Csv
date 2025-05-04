# Chat-With-Csv

A Streamlit app to chat with CSV files using a local Llama 2 model for private, offline data analysis.

## Features
- Upload CSV files and interact with your data using natural language.
- Powered by a locally running Llama 2 model (no cloud required).
- Uses LangChain for document loading, embeddings, and retrieval.
- All processing is done locally for privacy and speed.

## Getting Started
1. Clone the repository.
2. Download the Llama 2 model file (`llama-2-7b-chat.ggmlv3.q8_0.bin`) and place it in the project root.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## Notes
- The model file is **not included** in the repository due to its size.
- Add your model file to the project root as specified above.
- The `.gitignore` excludes model files, virtual environments, and vectorstore data.

## License
MIT
