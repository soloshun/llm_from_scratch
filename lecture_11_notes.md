# 🔹 **Positional Encoding in LLMs**

### 🧠 Why Positional Encoding?

Token embeddings alone do **not** capture the **order** of tokens.
For example, in both:

- `"Then the cat sat on the mat"`
- `"On the mat the cat sat"`

The token `cat` would be mapped to the **same vector**, even though its position changes. <br>
⚠️ This lack of positional awareness limits the model’s understanding of **syntax** and **contextual flow**.

![](images/L11_posi_enc.png)

### 📌 What is Positional Encoding?

It injects **positional information** into the token embeddings so the model can learn **token order**.

## 📘 Two Main Types

### 1. **Absolute Positional Encoding**

- Each **position** in the sequence has a **unique, learnable embedding vector**.
- These are **added** to token embeddings (same dimension as the embedding).
- Used in **GPT** and **original Transformers**.
- Works well for tasks where **fixed token order** is critical (e.g., sequence generation).

📊 Pros:

- Simple and effective for short-to-mid sequences.

⚠️ Cons:

- Harder to generalize to sequences longer than seen during training.

  ![](images/L11_abs_enc.png)

### 2. **Relative Positional Encoding**

- Focuses on the **relative distance between tokens** instead of exact positions.
- Model learns patterns like “this word is 2 tokens after the previous noun.”

📊 Pros:

- Better generalization across **variable-length** sequences.
- More flexible in long-context tasks.

⚠️ Cons:

- Harder to implement in vanilla transformers.

## 🏗 How Does This Help?

Adding positional encoding allows the LLM to:

- Understand **word order**
- Track **relationships between tokens**
- Improve **contextual predictions**

## 🛠️ Implementation in GPT

- Uses **absolute** positional encodings.
- Positional embeddings are **learnable parameters**.
- They are **optimized along with token embeddings** during training.

## 🔧 **Implementing Positional Embeddings (Hands-on)**

We walk through how **token embeddings** and **positional embeddings** are combined to create **input embeddings** in a real model pipeline.

## ![](images/L11_img1.png)

### 📐 Setup Parameters

| Hyperparameter  | Value          |
| --------------- | -------------- |
| `vocab_size`    | 50,257         |
| `embedding_dim` | 256            |
| `context_size`  | 4 (max_length) |
| `batch_size`    | 8              |
| `stride`        | 4              |

---

### 🧾 Step-by-step Breakdown

---

### **1. Get Input from DataLoader**

![](images/L11_img2.png)
Each training example gives:

- A sequence of **4 token IDs** per sample (because `context_size = 4`)
- Total shape: `8 x 4` (since `batch_size = 8`)

Each token ID is mapped to a **token embedding vector of size 256**, so:

- Token embeddings shape = `8 x 4 x 256` <br>
  (→ 8 samples, 4 tokens each, 256-dimensional vector per token)
  ![](images/L11_img3.png)

🧠 _Think of this as a 3D tensor: `Batch x Sequence Length x Embedding Dim`_

---

### **2. Create a Positional Embedding Layer**

- We need to represent **4 positions** → one for each token in the context window
- Shape = `4 x 256`
  (4 positions, each with a 256-dim vector)

📝 **Note**: This matrix is **independent of the actual tokens**. It's learned during training.

Since all examples in the batch share the same 4 token positions, this matrix is **reused** across the batch.

![](images/L11_img4.png)

---

### **3. Generate Positional Vectors**

- Extract 4 positional vectors (`0`, `1`, `2`, `3`) from the positional embedding matrix.
- These correspond to **positions within the context window**.

---

### **4. Add Token + Position Embeddings**

We combine:

```
Input Embeddings = Token Embeddings + Positional Embeddings
```

- Token Embedding Shape: `8 x 4 x 256`
- Positional Embedding Shape: `4 x 256`

🧠 **Broadcasting** happens here:

- The `4 x 256` positional embeddings are **broadcasted** (repeated) across the batch dimension to match `8 x 4 x 256`.

✅ Final input embedding passed to the model = `8 x 4 x 256`

---

### 💡 What is Broadcasting?

In NumPy/PyTorch:

- **Broadcasting** automatically expands tensors with smaller dimensions to match larger ones, as long as dimensions are compatible.
- Here, `4 x 256` is expanded into `8 x 4 x 256` to be added element-wise to token embeddings.

---

### ✅ Summary

| Component             | Shape         | Purpose                                    |
| --------------------- | ------------- | ------------------------------------------ |
| Token IDs             | `8 x 4`       | Indexes of words                           |
| Token Embeddings      | `8 x 4 x 256` | Converts tokens to dense vectors           |
| Positional Embeddings | `4 x 256`     | Encodes position info per token            |
| Input Embeddings      | `8 x 4 x 256` | Token + Position (combined input to model) |
