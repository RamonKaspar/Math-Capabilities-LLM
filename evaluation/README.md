# Evaluation

## Evaluation Terminology and Metrics

### Introduction

We deviate from using standard classification metrics such as True Positives (TP), False Negatives (FN), True Negatives (TN), and False Positives (FP) in our evaluation. This is because our dataset exclusively contains questions with a specific float as the correct answer; hence, there's no 'negative' class typically required for these traditional metrics. We also distinguish between incorrect answers and abstentions, where abstentions are a recognized outcome rather than incorrect or unattempted responses.

### Terminology

- **Correct Answer (CA)**: The technique returns the exact ground truth float.
- **Incorrect Answer (IC)**: The technique returns a float that is not correct.
- **Abstention (AB)**: The technique returns `None`, indicating it does not know the answer (or an error occured). While not ideal, this may be preferable to providing an incorrect answer.

### Metrics

- **Accuracy**: The proportion of all correct answers to the total number of questions. Abstentions are not counted as correct.
  $$\text{Accuracy} = \frac{\text{Number of Correct Answers}}{\text{Total Number of Questions}} = \frac{\text{CA}}{\text{Total Number of Questions}}$$

- **Precision**: - **Precision**: The proportion of correct answers among all the answers provided by the technique. This metric highlights the ability to avoid incorrect answers, with higher precision indicating a lower tendency to provide false information (i.e. a "hallucination").
  $$\text{Precision} = \frac{\text{Number of Correct Answers}}{\text{Number of Correct Answers + Number of Incorrect Answers}} = \frac{\text{CA}}{\text{CA + IA}}$$
