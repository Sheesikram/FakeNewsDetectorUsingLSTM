# Fake News Classifier Using LSTM

This repository contains code for a **Fake News Classifier** built using an LSTM (Long Short-Term Memory) model. The classifier identifies whether a given news article is real or fake based on its content. This project demonstrates the use of deep learning and Natural Language Processing (NLP) for text classification tasks.

## Project Overview

Fake news detection is a critical problem in today's world, where misinformation can spread rapidly through social media and other platforms. Using an LSTM-based neural network, this project trains a model to recognize patterns in text that indicate fake news, helping to improve content moderation and information verification.

## Features

- **Data Preprocessing**: The text data undergoes cleaning, tokenization, and padding.
- **LSTM Model Architecture**: A neural network that captures sequential information within text data.
- **Evaluation Metrics**: Measures the model's performance using accuracy, precision, recall, and F1-score.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.x
- TensorFlow or PyTorch (depending on the framework used)
- Jupyter Notebook (to view and run the `.ipynb` file)
- Additional libraries for data manipulation and visualization, such as:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/FakeNewsClassifierUsingLSTM.git
   cd FakeNewsClassifierUsingLSTM
   ```

2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook FakeNewsClassifierUsingLSTM.ipynb
   ```

3. **Run the Notebook Cells**: Execute each cell in the notebook sequentially to preprocess the data, train the model, and evaluate its performance.

## Model Architecture

The model architecture includes:
- **Embedding Layer**: Converts words into dense vectors.
- **LSTM Layers**: Captures the sequential dependencies in the text.
- **Dense Layers**: Fully connected layers that output a probability score for each class.

## Dataset

This project requires a dataset of labeled news articles (fake or real) to train the model. Common datasets used for fake news classification include:

- **Fake and Real News Dataset** (from Kaggle)
- **LIAR: A Benchmark Dataset for Fake News Detection**

Ensure the dataset is preprocessed into a suitable format (e.g., CSV file with columns for text and label).

## Results

The notebook includes steps to evaluate model performance on various metrics:
- **Accuracy**: The percentage of correct predictions.
- **Precision and Recall**: Measures for positive class predictions.
- **F1 Score**: Harmonic mean of precision and recall.

## Example

Below is an example usage of the classifier:
```python
# Assuming `model` is the trained LSTM model and `new_article` is a new input text
prediction = model.predict(new_article)
print("Fake News" if prediction == 0 else "Real News")
```

## Limitations

- **Data Dependency**: The accuracy and robustness of the model depend heavily on the quality and quantity of training data.
- **Computation Time**: LSTMs can be computationally expensive for large datasets.
- **Bias in Training Data**: If the dataset contains biases, the model may learn and replicate them.

## Future Improvements

Potential improvements include:
- Using more advanced models such as **BERT** or **Transformers**.
- Implementing a bidirectional LSTM for capturing context from both directions.
- Training on larger datasets to improve generalization.

## License

This project is licensed under the MIT License by Shees Ikram
