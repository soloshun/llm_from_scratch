# Stages of building LLMs

## Creating an LLM = Pretraining + Finetuning

### Pretraining

- Training on a large, diverse dataset
- sources of data for GPT3 are CommonCrawl, WebText2, Book1, Book2 and Wikipedia

### Finetuning

- refinement by training on narrower dataset, specific to a particular task or domain

## steps for building an LLM

1.  Train on a large corpus of text data(raw text)

    - Raw text = regular text without any labeling information

2.  first training stage of LLM is also called `pretraining`

    - creating an initial pretrained LLM(base/foundational model)
    - eg: GPT-3 model is a pretrained model which is capable of text completetion

3.  After obtaining the pretrained LLM, we can further train the LLM on labelled data

4.  Ther are 2 popular categories of finetuning

    - `instruction finetuning`: labeled dataset consist of instrution-answer pairs. eg: text translation, airline customer support
    - `fintuning for classification tasks`: labeled dataset consist of text and associated lables. eg: emails -> spam vs non-spam

5.  After finetuning deployment
