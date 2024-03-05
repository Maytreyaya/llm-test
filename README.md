# llm-test

The main purpose of this script is to answer users questions according to PDF files.

```shell
git clone https://github.com/Maytreyaya/llm-test
cd llm-test
python -m venv venv
venv\Scripts\activate (on Windows)
source venv/bin/activate (on macOS)
pip install -r requirements.txt
python main.py
To test the work of the script you should provide openai_api_key in .env file and full path to the file location to pdf_file_path variable.
```