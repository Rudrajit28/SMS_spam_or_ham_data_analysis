# 📩 SMS Spam Classifier (Machine Learning + Streamlit)

## 🚀 Overview

This project is a **SMS Spam Detection Web App** built using **Machine Learning** and deployed with **Streamlit**.
It classifies text messages as **Spam 🚫** or **Not Spam ✅** in real-time.

The model is trained on labeled SMS data and uses **Natural Language Processing (NLP)** techniques to understand and classify messages.

---

## 🧠 Features

* Classifies SMS messages instantly
* Clean and simple user interface
* Uses TF-IDF vectorization for text processing
* Machine Learning model for prediction
* Real-time interaction via web app

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas & NumPy
* NLTK (for preprocessing)
* Streamlit (for frontend)

---

## 📂 Project Structure

```
sms-spam-classifier/
│
├── app.py                # Streamlit application
├── model.pkl            # Trained ML model
├── vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
```

---

## ⚙️ How It Works

1. User enters a message in the web app
2. Text is preprocessed (lowercasing, stopword removal, stemming)
3. TF-IDF vectorizer converts text into numerical features
4. Model predicts:

   * `1` → Spam
   * `0` → Not Spam
5. Result is displayed on the screen

---

## 🧪 Example Inputs

### Spam

* "Congratulations! You won a free gift card"
* "Win cash now!!! Click here"

### Not Spam

* "Hey, are we meeting today?"
* "Call me when you're free"

---

## ▶️ Run Locally

### 1. Clone the repository

```
git clone <your-repo-link>
cd sms-spam-classifier
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run app.py
```

---

## 🌐 Deployment

This project can be deployed easily using **Streamlit Community Cloud**.

Steps:

1. Push code to GitHub
2. Go to Streamlit Cloud
3. Connect repository
4. Deploy `app.py`

---

## 📈 Future Improvements

* Add probability/confidence score
* Improve model accuracy with advanced algorithms
* Better UI/UX design
* Support for multiple languages
* Deploy with custom domain

---

## 🙌 Acknowledgements

* Dataset: SMS Spam Collection Dataset
* Libraries: Scikit-learn, NLTK, Streamlit

---

## 📌 Author

**Rudra**

---

## ⭐ If you like this project

Give it a star on GitHub and share it!
