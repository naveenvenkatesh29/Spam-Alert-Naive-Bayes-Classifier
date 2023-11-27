import os
import nltk
import codecs
import random
import logging
import smtplib
import gradio as gr
from nltk import Text
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from nltk import NaiveBayesClassifier, classify, word_tokenize

class SpamAlert():

    def __init__(self):

      """Class for spam detection and email filtering using Naive Bayes Classifier."""

      nltk.download("punkt")

      logging.basicConfig(filename='spam_alert.log', level=logging.INFO)

    def read_in(self, folder: str) -> list:

        """Read the content of files in the specified folder.

        Args:
            folder (str): Path to the folder containing files.

        Returns:
            list: List of document contents.
        """

        # Get the list of files in the specified folder
        file_list = os.listdir(folder)
        
        # Initialize an empty list to store document contents
        content_list = []

        # Iterate through each file in the folder
        for doc in file_list:

            # Ignore hidden files (those starting with dot)
            if not doc.startswith("."):

                # Open the document file for reading
                doc_read = codecs.open(folder + doc, mode="r", encoding="ISO-8859-1", errors="ignore")

                # Read the content of the document and append it to the list
                content_list.append(doc_read.read())

                # Close the file after reading
                doc_read.close()
        
        # Return the list of document contents
        return content_list

    def get_all_mails(self) -> list:

      """Get all emails from 'ham' and 'spam' folders.

      Returns:
          list: List of tuples containing email content and labels.
      """
      # Read contents of 'ham' folder
      ham_list = self.read_in("ham/")

      # Read contents of 'spam' folder
      spam_list = self.read_in("spam/")

      # Create a list of tuples, each containing email content and label ('ham')
      all_emails = [(email, "ham") for email in ham_list]

      # Extend the list with tuples for 'spam' emails
      all_emails += [(email, "spam") for email in spam_list]

      # Seed for reproducibility and shuffle the list
      random.seed(42)
      random.shuffle(all_emails)
      
      # Return the list of tuples containing email content and labels
      return all_emails


    def create_features(self, content: str) -> dict:

        """Create features from the given email content.

        Args:
            content (str): Email content.

        Returns:
            dict: Dictionary of features extracted from the content.
        """

        # Tokenize the email content into a list of words
        word_list = word_tokenize(content.lower())

        # Initialize an empty dictionary for features
        features = {}

        # Create features by setting each word in the content to True
        for word in word_list:
            features[word] = True

        # Return the dictionary of features
        return features

    def get_features(self) -> list:

        """Get features from all available emails.

        Returns:
            list: List of tuples containing features and labels.
        """

        # Get all emails with their labels
        all_emails =  self.get_all_mails()

        # Create a list of tuples containing features and labels
        return [(self.create_features(email), label) for (email, label) in all_emails]


    def train(self,proportion: float) -> tuple:

        """Train the Naive Bayes Classifier using a specified proportion of the dataset.

        Args:
            proportion (float): Proportion of the dataset to use for training.

        Returns:
            tuple: Tuple containing training set, test set, and the trained classifier.
        """
        try:
          # Get features from all available emails
          all_features = self.get_features()
          content = all_features

          # Determine the size of the training set based on the specified proportion
          sample_size = int(len(content) * proportion)

          # Split the dataset into training and test sets
          train_set = all_features[:sample_size]
          test_set = all_features[sample_size:]

          # Train the Naive Bayes Classifier
          classifier = NaiveBayesClassifier.train(train_set)

          # Return the training set, test set, and the trained classifier
          return train_set, test_set, classifier

        except Exception as e:
            # Log any exceptions during training
            logging.error(f"Error during training: {e}")
            raise

    def evaluate(self) -> NaiveBayesClassifier:
        
        """Evaluate the performance of the trained classifier on the test set.

        Returns:
            NaiveBayesClassifier: Trained Naive Bayes Classifier.
        """
        try:
          # Train the classifier using 80% of the dataset
          train_set, test_set, classifier = self.train(.80)

          # Print accuracy for the training set
          print(f"Accuracy for train set: {classify.accuracy(classifier, train_set)}")

          # Print accuracy for the test set
          print(f"Accuracy for test set: {classify.accuracy(classifier, test_set)}")

          # Show the most informative features
          NaiveBayesClassifier.show_most_informative_features(classifier)

          # Return the trained classifier
          return classifier

        except Exception as e:
            # Log any exceptions during evaluation
            logging.error(f"Error during evaluation: {e}")
            raise

    def concordance(self, data_list: list, search_word: str) -> None:

        """Print concordance lines for the specified search word in each document.

        Args:
            data_list (list): List of documents.
            search_word (str): Word to search for in the documents.
        """

        # Iterate through each document in the list
        for data in data_list:

            # Tokenize the document into a list of words
            word_list = word_tokenize(data.lower())

            # Create a Text object from the word list
            text_list = Text(word_list)

            # Check if the search word is present in the document
            if search_word in text_list:

                # Print concordance lines for the search word
                text_list.concordance(search_word)

    def send_email(self, subject: str, message_body: str, recipient_email: str) -> str:
        
        """Send an email using SMTP.

        Args:
            subject (str): Email subject.
            message_body (str): Email body.
            recipient_email (str): Recipient's email address.

        Returns:
            str: Success or error message.
        """

        # Replace these with your own email and SMTP server details
        sender_email = 'YOUR_EMAIL_ID'
        sender_password = 'YOUR_APP_PASSWORD'
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587  # 587 is the default port for TLS, use 465 for SSL

        try:
            # Create a connection to the SMTP server
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)

            # Create an email message
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = recipient_email
            message['Subject'] = subject

            # Add the email body
            message.attach(MIMEText(message_body, 'plain'))

            # Send the email
            server.sendmail(sender_email, recipient_email, message.as_string())

            return "Email sent successfully!"

            # Close the SMTP connection
            server.quit()
        except Exception as e:

            return f"Email sending error: {e}"

    def message_alert(self, final: str, email: str) -> None:

        """Send a message alert email based on the classification result.

        Args:
            final (str): Classification result (spam or ham).
            email (str): Recipient's email address.
        """

        # Set the subject for the alert email
        subject = "Message Alert"

        # Create the message body based on the classification result
        message_body= f"You Received {final} Message"

        # Use the 'send_email' method to send the alert email
        self.send_email(subject, message_body, email)

    def get_email(self, email: str, subject_email: str) -> str:

        """Get an example email.

        Args:
            email (str): Recipient's email address.
            subject_email (str): Email subject.

        Returns:
            str: Success or error message.
        """

        # usage:
        subject = "Test Email Subject"
        message_body = subject_email
        recipient_email = email

        # Call the 'send_email' method to send the example email
        return self.send_email(subject, message_body, recipient_email)

    def filter_email(self, email: str, subject: str) -> tuple:

        """Filter the input email using the trained classifier.

        Args:
            email (str): Recipient email address.
            subject (str): Email subject.

        Returns:
            tuple: Tuple containing the email sending status and classification result.
        """
        try:
          # Evaluate the trained classifier
          classifier = self.evaluate()

          # Get an example email
          message = self.get_email(email,subject)

          # Classify the subject using the trained classifier
          final = classifier.classify(self.create_features(subject))

          # Send a message alert based on the classification result
          self.message_alert(final,email)

          # Return a tuple containing the email sending status and classification result
          return message,classifier.classify(self.create_features(subject))
        except Exception as e:

            # Log any exceptions during email filtering
            logging.error(f"Error during email filtering: {e}")
            raise

    def gradio_interface(self) -> None:

        """Launch the Gradio interface for email filtering."""

        with gr.Blocks(css="style.css",theme= "abidlabs/pakistan") as demo:
            gr.HTML("""<center class="darkblue" text-align:center;padding:25px;'>
            <h1 style="color:#fff">
                📧Spam filter🚫
            </h1>
            </center>""")
            # gr.HTML("""<center><b><h1 style="color:#fff">📧Spam filter🚫</h1></b></center>""")
            with gr.Column(elem_id="col-container"):
              with gr.Row():
                with gr.Column(scale=0.8):
                  email = gr.Textbox(placeholder="Enter E-Mail ID", label="E-Mail ID")
                with gr.Column(scale=0.2,min_width=160):
                  check = gr.Button(value="Submit",elem_classes="submit")
              with gr.Row():
                email_sub = gr.TextArea(max_lines=20, placeholder="Enter email Here", label="E-Mail")
              with gr.Row():
                email_success = gr.Label(label="Status")
              with gr.Row():
                out = gr.Label(label="Result")

              check.click(fn=self.filter_email, inputs=[email,email_sub], outputs=[email_success,out])
        demo.launch(debug = True)
