# SPAM SMS-MAIL CLASSIFIER

Dataset: `spam.csv` from Kaggle

Algorithm: `Naive Bayes Classifier`

Screenshots:

## ![Screenshot 2024-09-28 105931](https://github.com/user-attachments/assets/a3719796-1905-4f4d-8374-ed1bb5015ec0)

![Screenshot 2024-09-28 110004](https://github.com/user-attachments/assets/8b093b87-5349-4858-be2c-f64fdc55cb1a)

# Flow of the Model

### Importing Libraries

1. The program begins by importing necessary libraries like ```numpy```, ```pandas```, and ```scikit-learn``` modules.
2. ```Pandas``` is used for data manipulation, while ```scikit-learn``` is used for machine learning.

### Loading Data

1. The program loads a dataset `spam.csv` using `pandas.read_csv()`. The data likely consists of text messages and labels `spam` or `ham`.

### Data Preprocessing

1. Columns that are not necessary (like unnamed columns) are dropped from the dataset.
2. The columns are renamed: one column is renamed to ```target``` (indicating spam or ham), and other column is renamed to ```text```.
3. The target labels are encoded using ```LabelEncoder``` where the classes ```spam``` and ```ham``` are transformed into binary values ```0``` and ```1```.

### Handling Missing or Duplicate Data

1. The program checks for missing values using ```isnull().sum()``` and counts the number of duplicates using ```df.duplicated().sum()```.
2. Duplicates are dropped, keeping only the first occurrence.

### Text Preprocessing

1. Text data is preprocessed by converting it into lowercase, tokenizing, removing special characters, and stopwords.
2. The text data is then transformed into vectors using TF-IDF (Term Frequency-Inverse Document Frequency) via ```TfidfVectorizer```. This converts the text into a numerical format suitable for machine learning models.

### Splitting Data

1. The dataset is split into training and testing sets using ```train_test_split()```. A portion(80%) of the data is used for training, and the remaining 20% is used for testing the model.

### Model Training

1. Three Naive Bayes classifiers are instantiated: ```GaussianNB```, ```MultinomialNB``` & ```BernoulliNB```

2. Each of these classifiers is trained on the training data using ```fit()```.

### Model Evaluation

1. After training, the program evaluates each classifier on the test data using: 
```Accuracy Score```(Percentage of correct predictions)
```Confusion Matrix```(A matrix summarizing true positives, false positives, true negatives, and false negatives.)
```Precision Score```(Measures the precision of the classifier for spam prediction.)

2. These metrics help compare the performance of the classifiers.


### Saving the Model

1. After the evaluation, the program saves the trained model ```MultinomialNB``` and the vectorizer ```TfidfVectorizer``` using the ```pickle``` library. 
2. These saved models can be loaded later for real-time spam classification.


# Why This Approach Was Used?

1. ```Naive Bayes Classifiers```: Naive Bayes is ideal for text classification problems due to its simplicity, effectiveness, and speed, especially when dealing with large datasets like text.

2. ```TF-IDF Vectorization```: Converting text data into numerical form using TF-IDF is a common approach in Natural Language Processing (NLP). It helps to capture the importance of words in relation to their frequency in the text.

3. ```Model Comparison```: By training multiple Naive Bayes models, the program ensures the best-performing model is chosen based on accuracy and precision metrics.


This structure allows for effective detection of spam in text data with high accuracy and precision, particularly using the MultinomialNB classifier which works well for discrete features like word counts.