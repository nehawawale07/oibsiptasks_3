# Email Spam Detection System

## Overview
The **Email Spam Detection System** is a machine learning-based project designed to classify emails as spam or ham (not spam). The system uses natural language processing (NLP) techniques and supervised learning algorithms to analyze email content and detect spam with high accuracy.

## Features
- Preprocessing of email text data (tokenization, stopword removal, stemming, etc.)
- Implementation of various machine learning models (Naïve Bayes, SVM, Decision Tree, etc.)
- Performance evaluation using precision, recall, and F1-score
- Interactive interface for testing emails
- Dataset support and model training pipeline

## Technologies Used
- Python
- Scikit-learn
- Natural Language Toolkit (NLTK)
- Pandas & NumPy
- Flask (if a web interface is included)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nehawawale07/email-spam-detection.git
   cd email-spam-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the system:
   ```bash
   python main.py
   ```

## Usage
- **Training:**
  ```bash
  python train.py
  ```
- **Testing with Sample Emails:**
  ```bash
  python test.py --email "Your sample email content here"
  ```
- **Web Interface (if applicable):**
  ```bash
  flask run
  ```

## Performance
- Accuracy: XX%
- Precision: XX%
- Recall: XX%
- F1-score: XX%

## Contributing
Feel free to fork this repository, create new branches, and submit pull requests for improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any queries or suggestions, contact [wawaleneha07@gmail.com).

