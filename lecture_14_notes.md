# A simplied self attention mechanism without trainable weights

- The goal for attention is pick the `vector + positions embeddings` which captures the semantic meaning and others... and `outputs a context vector`
- the `context vector` can be considered as an enriched `embedding vector` because it not only contains the semantic meaning but also contains information on how each word/token in the input sequence/sentence relate to each other. and this context helps model make the right prediction
  ![](images/L14_atten.png)

## The goal of self attention

input, x: "your journey starts with one step"

- "Calculate a context vector `z(i)` for each element in `x(i)`"

  - Context vector: enriched `embedding vector`

- we find the context vector for each embedding element.

  - so we focus on `x(2)-query` for now with it coresponding context vector `z(2)`.
  - `z(2)` is an embedding which contains information about `x(2)` and all other input element `x(1)` to `x(T)`

- in this lecture we do not add the trainable weights. the trainable weights help the LLM learn to construct these context vectors, so that they relevant for the LLM to generate the next token

so our task to it convert the embeddings vector of a `query` into a context vector

1. First step of implementing self-attention is to compute the intermediate values `w`, also referred to as `attention scores`
   ![](images/L14_s1.png)

2. The intermediate `attention scores` are calculated between the `query token`(dot product between query and every other input token) and each `input token` us the `dot product`.

   - dot products quantifies how much two vectors are aligned
   - In the context of self attention mechanisms, **dot products determines the extent to which elements of a sequence attend to one another**, the higher the dot product, the higher the `similarity` and `attention score`

3. Normalize Attention Scores

   - While a simple normalization (dividing each score by the sum of all scores) is possible, it doesn't handle extreme values well. For instance, with scores like `[1, 3, 4, 400]`, the largest value would dominate, but smaller values would still retain some weight. This can be problematic during training because even small, near-zero weights can receive undue attention during backpropagation, potentially confusing the model.
   - A better approach is to use the `softmax` function. Softmax amplifies the highest scores and diminishes the smaller ones, making the highest score's weight approach 1 while the others become negligible. This helps the model focus on the most relevant tokens.
   - The standard softmax formula is shown below:
     ![](images/L14_s2.png)
     ![](images/L14_s2_5.png)
   - However, this naive implementation can be numerically unstable when dealing with very large or small numbers, leading to overflow or underflow errors. To address this, a common technique is to subtract the maximum value from each score before applying the exponentiation. This is a more stable implementation, as used in libraries like PyTorch.
     ![](images/L14_s3.png)

4. Computing the context vector
   - after computing the normalized `attention weights`, we calculate the `context vector z(z)` by multiplying the `embedding input tokens x(i)`, with the coresponding `attention weights` and then summing the resultant vectors
   - after computing the normalized attention, **we multiply each of the input embedding vector by the corresponding attention weight to scale them down, then take the vector sum of them and that gives the context vector**
     ![](images/L14_s4.png)
     ![](images/L14_s4_5.png)
     ![](images/L14_s5.png)

- attention weight matrix

  NB: apart from meaning, it is also important to capture, you also need to capture the information of the context and without trainable weights we'll not be able to capture context correctly
