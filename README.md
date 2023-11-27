# Spam-Alert-Naive-Bayes-Classifier
The SpamAlert project is a Python-based email filtering system that uses a Naive Bayes Classifier to classify emails as either spam or ham (non-spam). The system is implemented as a class named SpamAlert, which provides methods for training the classifier, evaluating its performance, and filtering emails based on the trained model.

# Features

- **Email Content Reading:** The system reads the content of emails from specified 'ham' and 'spam' folders.

- **Feature Extraction:** It tokenizes the email content and creates features for the Naive Bayes Classifier.

- **Naive Bayes Classifier Training:** The system trains the Naive Bayes Classifier using a specified proportion of the dataset.

- **Classifier Evaluation:** It evaluates the performance of the trained classifier on a test set and prints accuracy metrics.

- **Concordance:** The system can print concordance lines for a specified search word in each document.

- **Email Sending:** It provides the functionality to send emails using SMTP with specified email subject, body, and recipient.

- **Message Alert:** The system sends a message alert email based on the classification result (spam or ham).

- **Gradio Interface:** It launches a Gradio interface for email filtering, allowing users to input an email and subject for classification.

# Usage

**Initialize the SpamAlert Class:**

      spam_alert = SpamAlert()

**Train the Naive Bayes Classifier:**

      train_set, test_set, classifier = spam_alert.train(proportion=0.8)

**Evaluate the Classifier:**

      trained_classifier = spam_alert.evaluate()

**Filter Email:**

      message, classification_result = spam_alert.filter_email(email="recipient@example.com", subject="Test Subject")

**Gradio Interface:**

      spam_alert.gradio_interface()

# Dependencies

- os: Operating system interfaces
- nltk: Natural Language Toolkit for text processing
- codecs: Codec registry and base classes
- random: Generate pseudo-random numbers
- logging: Logging facility for Python
- smtplib: Simple Mail Transfer Protocol client
- gradio: Gradio for creating user interfaces
- email.mime: MIME library for creating email messages

# Configuration

- Folders: The 'ham' and 'spam' folders should contain the training dataset.
  
- SMTP Configuration: Replace YOUR_EMAIL_ID, YOUR_APP_PASSWORD, smtp_server, and smtp_port with your own email and SMTP server details

# Note

- Ensure that the necessary datasets are available in the 'ham' and 'spam' folders.
- Replace the SMTP configuration details with your own email credentials.


# Contact

If you have any doubt about this github feel free and ask: Email : naveenvenkateshkumar@gmail.com
