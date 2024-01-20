import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Load data
data = pd.read_csv("SMSSpamCollection.csv", names=["Label", "message"], encoding="latin-1")

# Text preprocessing
lemmatizer = WordNetLemmatizer()
data['message'] = data['message'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in re.sub('[^a-zA-Z]', ' ', x).lower().split() if word not in set(stopwords.words('english'))]))

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(data['message'], data['Label'], test_size=0.33, random_state=42)

# Model training
text_svm = Pipeline([('tfidf', TfidfVectorizer()), ('svm', LinearSVC())])
text_svm.fit(X_train, Y_train)

# Kivy App
class SpamDetectorApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        self.message_input = TextInput(multiline=True, hint_text="Enter the message")
        self.result_label = Label(text="Result: ")

        classify_button = Button(text="Classify", on_press=self.classify_message)
        self.layout.add_widget(self.message_input)
        self.layout.add_widget(classify_button)
        self.layout.add_widget(self.result_label)

        return self.layout

    def classify_message(self, instance):
        message = self.message_input.text
        if message:
            refined_message = ' '.join([lemmatizer.lemmatize(word) for word in re.sub('[^a-zA-Z]', ' ', message).lower().split() if word not in set(stopwords.words('english'))])
            prediction = text_svm.predict([refined_message])
            result_text = f"Result: {prediction[0]}"
            self.result_label.text = result_text
        else:
            self.show_popup("Error", "Please enter a message.")

    def show_popup(self, title, content):
        popup_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        popup_layout.add_widget(Label(text=content))
        popup = Popup(title=title, content=popup_layout, size_hint=(None, None), size=(300, 200))
        popup.open()

if __name__ == '__main__':
    SpamDetectorApp().run()
