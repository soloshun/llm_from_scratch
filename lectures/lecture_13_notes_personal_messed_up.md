# Attention mechanism introduction

[get full animation here and article here of how attention works](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

[[1706.03762] Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## the 4 types of attention mechanisams:

- Simplied Self-attention
- Self-attention
- casual attention
- multi-head attention(gpt uses this)

![](../images/L13_types_att.png)

### Simplied Self-Attention

- A simplied and puriest self attention technique.

### Self Attention

- Self-attention with `trainable weights`, that form the basis of mechanism used in LLMs

### Causal Attention

- A type of self-attention used in LLMs that allows the model to only consider previous and current input in a sequence

### Multi-headed attention

- An extension of `self-attention` and `causal attention` that enables the model to simultaneously attend to information from different representation subspaces(LLM attends to various input data in parallel)

## The problem with modeling long sequences:

- What is the problem with architectures without the attention mechanism which came before LLMs?

  - **let's consider a language translation model**
    ![](../images/L13_s1.png)

    > **Word by word translation does not work!** > ![](../images/L13_s2.png) > ![](../images/L13_s3.png) > **The translation process requires `contexual understanding` and `grammer alignment`**

  - To address this issue that we cannot translate text word by word, it is common to use a neural network with two submodules:
  - **Encoder:** Reads and process text
  - **Decoder:** Translate text

    [get full animation here and article here](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
    ![](../images/L13_s4.png)
    [watch video](../images/L13_enc_dec1.mp4)

    > Under the hood, the model is composed of an `encoder` and a `decoder`. <br> <br> The `encoder` processes each item in the input sequence, it compiles the information it captures into a vector (called the `context`). After processing the entire input sequence, the `encoder` sends the `context` over to the `decoder`, which begins producing the output sequence item by item.

  - Before transformers, Recurrent Neural Networks(RNNs) were the most popular encoder-decoder architecture for language translations
  - RNN: output from previous step is fed as input to the current text
  - rnn takes the input and hidden state vector this hidden state vector gets updated at each iteration of every word
  - Here's how the encoder-decoder RNN works:
    ![](../images/L13_s5.png)

    - Input text
    - Encoder processes input sequencially
    - Updates `hidden state` at each step(internal values at hidden layers)... get updated at each iteration.
    - final hidden state(encoder tries to capture sentence meaning)
    - decoder uses the final hidden state to generate the translated sentence one word at a time(decoder also updates it's hidden state at each time)

  > (what i understood) okay this is an issue here is that the decoder doesn't have acess to all the hidden state from encoder, it only has the final last hidden state to work. so the decoder somewhat loses context of what all the encoder did and only has access to the final hidden state which is the summery of all the context the encoder as learnt... a senerio here... think of it like being asked to write and exam for course that has been studied over 6months. and they just give you the summery of all the 6months studies to write the exam on... you did realize so much context will be lost here.. and even somethings wouldn't even make sense. lot of pressure on the decoder...
  > **actual note from lecture continous**

  - The encoder process the entire input text into a hidden state(think of this as an embedding vector) (memory cell). the decoder takes this hidden state to produce an output

  - Big issue:
    1. RNN can't directly access earlier hidden states from the encoder, during the decoding phase.
    2. it relies solely on current hidden state
    3. this leads to a loss of context, especially in complex sentenses where dependencies might span long distances
       > Encoder compreses entire input sequence into a single hidden state vector. if the sentence is very long, it becomes very difficult for the RNN to capture all information in single vector.
       > eg: "The cat that was sitting on the mat, which was next to the dog, jumped" -> "please covert french version here pleaes"
       - here the key action `jumped` depends on the subject `cat` but also on understanding the `longer dependencies`("the was sitting on the mat, next to the dog")
       - the RNN decoder might struggle with this... lost of context

## Capturing data dependencies with attention mechanism

- RNNs works fine for translating short sentences, but don't work for long texts as they don't have direct access to previous words in the input
- One major shortcoming in this approach is that: **RNN must remember the entire encoded input in a single hidden state before passing it to the decoder**... _(ohh my thought here... somehing clicked) ooohhh that is why the embedding and positional embedding makes sense here case with the vector and positional embeddings which helps us understand the semantic meaning of words in vector spaces and wher each word is important here or not or something. the attention using these will actually be able some what decrypt how they are close in vector space or something... (i actually do not know my that is what i think... all is coming together here)_
- In 2014, researchers developed the so called **"Bahdanau attention mechanism for RNNs": `modifies` the `endcoder-decoder RNN` such that `decoder can selectively access different part of the input sequence at each decoding step`** [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE paper](https://arxiv.org/abs/1409.0473)
  ![](../images/L13_s6.png)

  - using an attention mechanism, the text generating decoder part of the network can access all input tokens selectively.
  - **this means that some input tokens are more important than other for generating a given output token.**
  - this importnce is determined by the so called `attention weights`

- Only 3 years later, researchers found that RNN architectures are not required for building deep neural networks for natural language processing and proposed the original transformer architecture; with a self-attention mechanism inspired by the `Bahdanau attention` mechanism

  > using "The cat that was sitting on the mat, which was next to the dog, jumped" -> "please covert french version here please" as an example,

  - what the attention mechanism does is that, `at each decoding step`, `the model can look back at the entire input sequence` and `decide which parts are most relevant to generate current word`
  - when the decoder is predicting `saite`(french of course), the attention mechanism allows it to focus on part of input that corresponds to `jumped`
  - `dynamic focus`(for every decoding step we can selectively choose which input to focus and how much attention we give to it) on different parts of the inputs sequence allows the model to learn long range dependencies more effectively

    > Note that the model isn’t just mindlessly aligning the first word at the output with the first word from the input. It actually learned from the training phase how to align words in that language pair (French and English in our example). An example for how precise this mechanism can be comes from the attention papers listed above:
    > ![](../images/L13_s7.png)

    > history of language models. we only hear of transformers today but, this work has actually been under research for the past 43 years
    > ![](../images/L13_s7_his.png)

- Self attention is a mechanism that allows each position of the input sequence to attend to all positions in the same sequence when computing the representation of a sequence.
- Self attention is a key component of the contemporary LLMs based on the transformer architecture, such as GPT series
  ![](../images/L13_s8.png)

## Attending to different parts of the input with self attention

- In `self attention`, the `self` refers to the mechanisms' ability to compute attention weight by relating diffent positions in a single input sequence.
- It learns the relationship between various parts of the input itself, such as words in a sentence.
- this is in contrast to tradtional attention mechanisms where the focus is on relationship between elements of 2 different sequences
