# Dataset Considerations

This README is about data set properties that should be
considered before new data is added.

## Reproducibility
* [ ] Can the dataset be programmatically reproduced ?
  * [ ] Is it relying on any manual work (lab included)?

A software-generated dataset is favourable since it can
be reproduced easily by another party, which helps with
collaborative development and comparative studies.

## Data Skew and Demographics
* [ ] Is the dataset representative of the expected patient
  group?
* [ ] Data skew; how is the data produced? For what purpose.
  Is the production method fit for your use-case?
  In what way is the data skewed?

## Model Overfitting
* [ ] Is the data set used somewhere else (other projects, software tools)
  that's used by our analysis pipeline? If so, there's a risk of overfitting
  to features that make use of this particular data set since the feature
  will perform (hopefully) well on the particular data.
* [ ] Consider dataset circularity.

## Team Knowledge
* [ ] Is the data set known by the team? If so, the previous knowledge
  might help in debugging performance bottlenecks.
