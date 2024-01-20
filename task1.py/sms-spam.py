import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, "SMSSpamCollection.csv")
data = pd.read_csv(file_path , names= ["Label" , "message"], encoding="latin-1")
data['Label'].value_counts()
message_len = 0
length = []
for i in range(len(data)):
    message_len = len(data['message'][i])
    length.append(message_len)
data['length'] = length
count = 0
punct = []
for i in range(len(data)):
    for j in data['message'][i]:
        if j in string.punctuation:
            count += 1
    punct.append(count)
    count= 0
data = data.assign(punct=punct)
# nltk.download('stopwords')
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(0, len(data)):
    words = re.sub('[^a-zA-Z]', ' ', data['message'][i])
    words = words.lower()
    words = words.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    words = ' '.join(words)
    
    corpus.append(words)
data['message'] = corpus
spam_messages = data[data['Label'] == 'spam']
ham_messages = data[data['Label'] == 'ham']
X = data['message']
Y = data['Label']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)
Tfidf = TfidfVectorizer()
count_vect=CountVectorizer()

X_train_count_vect=count_vect.fit_transform(X_train).toarray()
text_mnb=Pipeline([('tfidf',TfidfVectorizer()),('mnb',MultinomialNB())])
text_mnb.fit(X_train,Y_train)
y_preds_mnb=text_mnb.predict(X_test)
text_svm=Pipeline([('tfidf',TfidfVectorizer()),('svm',LinearSVC())])
svm_model = SVC()
text_svm.fit(X_train,Y_train)
y_preds_svm=text_svm.predict(X_test)
def refined_text(text):
    #Removal of extra characters and stop words
    words = re.sub('[^a-zA-Z]',' ',text)
    words = words.lower()
    #Splits into list of words 
    words = words.split()

    #Lemmatizing the word and removing the stopwords
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]

    #Again join words to form sentences
    words = ' '.join(words)
    return words

# Function to classify the message and display the result
def classify_message():
    user_input = input_text.get(1.0, tk.END)
    refined_word = refined_text(user_input)

    prediction = text_mnb.predict([refined_word])[0]

    if prediction == 'spam':
        result = "This message is spam."
    else:
        result = "This message is not spam."

    messagebox.showinfo("Spam Classifier Result", result)

# Create Tkinter window
window = tk.Tk()
window.title("Spam Classifier")
window.geometry("800x800")

# Create text input box
input_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=40, height=10)
input_text.pack(pady=10)

# Create classify button
classify_button = tk.Button(window, text="Classify", command=classify_message)
classify_button.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()