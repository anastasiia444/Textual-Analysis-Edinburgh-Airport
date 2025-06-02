README - Usage Instructions

1) Open a terminal and move into the project folder (the one containing NLP_Comments.py):

   cd path/to/your/project/folder

   Example (if it's on Desktop):
   cd ~/Desktop/YourProjectFolder        # on Mac/Linux
   cd C:\Users\YourName\Desktop\YourProjectFolder   # on Windows

2) Activate the virtual environment:
   - On Mac/Linux:
       source env/bin/activate
   - On Windows:
       env\Scripts\activate

3) Install missing libraries (if needed):
   pip install streamlit
   pip install torch        
   pip install openpyxl     # required to read Excel files

4) Run the app:
   streamlit run NLP_Comments.py

5) If you get an error related to NLTK, run this once:
   python
   >>> import nltk
   >>> nltk.download('punkt')
   >>> nltk.download('stopwords')
   >>> exit()

6) Make sure the file 'customer_responses_2025.xlsx' is in the same folder as NLP_Comments.py.

7) Once launched, the app should open in your browser at:
   http://localhost:8501

