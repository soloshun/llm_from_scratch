# Self attentions(scaled dot product attention) with trainable weights

In this lecture, we will learn about the `self-attention mechanism` used in the original transformer, the GPT models and other popular LLMs

This self attention mechanism also called `scaled dot-product attention`
![](images/L15_s1.png)

**Objectives of this lecture**

1. Compute context vectors as weighted sums over the input vectors specific to a certain input element
2. Introduce weight matrics that are updated during model training
3. These trainable weights matrics are crucial so that the model can learn to produce `good` context vectors
4. Implement the self-attention mechanism step by step by introducting 3 trainable weight: `Wq`, `Wk` and `Wv`
   - These 3 matrics are used to project the embedded input token `x(i)` into `query`, `key` and `value` vectors
     ![](images/L15_s3.png)

> Just as a reminder, the context vector is taught of as the `enriched embeddings`, it not only contains information and semantic meaning about each token, but also contains information about how each and every embeddings relates to each other

> The goal is still the same as implemented in the lecture 14... we want to get from `input embeddings` to `context vectors`

### **Steps 1: Converting input embeddings into `query`, `key`, `value` vectors**

![](images/L15_s2.png)

- so we first multiply the `input embeddings vectors` by a randomized vectors for the `query`, `key` and `value`. these are the trainable weights. so we'll have
  - `inputs * wq` -> `queries`
  - `inputs * wk` -> `keys`
  - `inputs * wv` -> `values`
- henceforth after we get the queries, keys and values we're not going to look at the inputs embeddings again. they've been transformed into 3 matrixs

### **Steps 2: Computing attention scores**

![](images/L15_s4.png)

### **Steps 3: Computing attention weights**

- this is done by normalizing the attention scores
- this ensures that all the values are between 0-1 and can be compared to see which token should be payed more attention when predicting
- before softmax is computed, the attention scores are scaled by the `square root of d-keys`. `d-keys` = dimension of the keys matrix.
- this is the reason why it is called `scaled dot product attention` because the attention score are being scaled by dividing the attention scores with the square root of the dimension of the keys matrix
- the scaling is done before applying softmax on the attention score to get the attention matrix

![](images/L15_s5.png)

### **Steps 4: Computing context vector**

- the context vector is computed by multiplying the attention weights withe values weights matrix. `context vector = attention weight * values`

![](images/L15_s6.png)
![context vector intuition](images/L15_s7.png)

![overview of the self-attention process](images/L15_s8.png)

### Analogy behind Query, Key, and Value in Self-Attention

- **Query (Q):**  
  Think of a **search request**. It represents what the current token is "asking" or "looking for" in the sequence.

- **Key (K):**  
  Think of an **index entry** in a database or a label on a filing cabinet. Each token has a key, which is what the query compares itself against to decide relevance.

- **Value (V):**  
  Think of the **information stored** behind the index. Once we know which keys match the query best, we pull out the corresponding values as the actual content.

---

### Scenario 1: Search Engine

- **Query:** The words you type into Google (e.g., "best Italian restaurants").
- **Keys:** Keywords/index entries of all web pages.
- **Values:** The actual web pages or snippets you get back.
- **Attention =** The query matches strongly with some keys → return their values.

---

### Scenario 2: Library

- **Query:** You walk in asking, "Books about World War II history?"
- **Keys:** Catalog entries (title, author, subject tags).
- **Values:** The actual books on the shelves.
- **Attention =** Librarian looks at catalog (keys), finds relevant matches, then hands you the books (values).

---

### Scenario 3: Group Conversation

- **Query:** You ask in a group, "Who remembers what the plan was for tomorrow?"
- **Keys:** Each person’s memory tags (who remembers what).
- **Values:** The actual information they recall and share.
- **Attention =** Your question (query) aligns most with the people (keys) who have that info → they give you the answer (values).

---

✅ **Key insight:**

- Queries decide _what I want_.
- Keys represent _how things are labeled_.
- Values contain _the substance to return once relevance is found_.

This is why in LLM self-attention:

- Each token generates its own query, key, and value.
- Tokens "ask" (query), compare against others (keys), and collect information (values) → leading to contextual understanding.
