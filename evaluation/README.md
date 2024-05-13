# Evaluation

## Evaluation Terminology and Metrics

### Introduction

We deviate from using standard classification metrics such as True Positives (TP), False Negatives (FN), True Negatives (TN), and False Positives (FP) in our evaluation. This is because our dataset exclusively contains questions with a specific float as the correct answer; hence, there's no 'negative' class typically required for these traditional metrics. We also distinguish between incorrect answers and abstentions, where abstentions are a recognized outcome rather than incorrect or unattempted responses.

### Terminology

- **Correct Answer**: The technique returns the exact ground truth float.
- **Incorrect Answer**: The technique returns a float that is not correct.
- **Error**: The technique returns `None`, indicating that an error occured. While not ideal, this may be preferable to providing an incorrect answer.

### Metrics

**Accuracy**: The proportion of all correct answers to the total number of questions. Errors are counted as incorrect.
$$\text{Accuracy} = \frac{\text{Number of Correct Answers}}{\text{Total Number of Questions}}$$
