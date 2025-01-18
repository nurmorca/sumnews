# News Summarization with T5, PEGASUS, and mT5

This project implements and fine-tunes state-of-the-art language models, including **T5 Base**, **PEGASUS**, and **mT5**, for news summarization tasks. The project specifically focuses on summarizing news articles from **CNN** and **TRT Haber**, with a particular emphasis on summarizing Turkish news using mT5. 

## Features

- **News Gathering**: Scrapes and processes news articles from the CNN and TRT Haber websites.
- **Fine-Tuning**: Custom fine-tuning of T5 Base, PEGASUS, and mT5 for optimal summarization performance.
- **Language Support**: Handles English news (CNN) and Turkish news (TRT Haber).
- **Evaluation**: Includes metrics to evaluate the summarization quality.
- **Web UI**: Users can view summarized news through a web page, powered by Streamlit.

## Models Used

1. **T5 Base**: A transformer model pre-trained on a diverse set of text-to-text tasks.
2. **PEGASUS**: A transformer model designed for abstractive summarization tasks.
3. **mT5**: A multilingual version of T5, supporting Turkish and other languages.

## Installation

### Prerequisites

- Python 3.8+
- Git
- Pip

### Clone the Repository
```bash
git clone https://github.com/nurmorca/sumnews
cd sumnews
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
Run the application:
```bash
streamlit run path/to/main.py
```


## Project Structure

```
sumnews/
├── Code/                 # Folder for all the code in the project, including for fine-tuning and UI.
├── Database/             # Folder for the .txt file that keeps the info of the datasets that's been used for fine-tuning.
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
```

## Acknowledgments

- **CNN** and **TRT Haber** for providing news articles.
- Hugging Face Transformers library for pre-trained models.
- Hugging Face users for datasets!

---
Feel free to raise issues or contribute to this repository for any enhancements or bug fixes.

