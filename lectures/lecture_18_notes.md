# Multi-Head Attention(Part 2): Implementing Multi-Head Attention with weight splits

- Instead of maintaining 2 seperate classes: `MultiHeadAttentionWrapper` and `CausalAttention` as done in [lecture 17 notebook](lecture_17.ipynb), we combined both of these into a single `MultiHeadAttention` class

  - comparing btn multi-head attention `by contactenation` and `with weight split` image below
    ![comparing btn multi-head attention `by contactenation` and `with weight split image`](../images/L18_s1.png)

    - number of attention head is specified. `Head dimension = d_out/n_head`

### **We start with a simple example from scratch:**

1. **Step 1:** Start with the input `b, num_tokens, d_in = (1, 3, 6)`

   ```python
   x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])
   ```

   - `batch = 1`, `num_tokens = 3`, `d_in = 6`

2. **Step 2:** Decide `d_out = 6`, `num_head = 2`. so now each head dimension will no be `head_dim=d_out/num_heads` => `head_dim=6/2=3`

3. **Step 3:** Initialise trainable weight matrices for `query`, `key` and `value`(Wq, Wk, Wv).

   - the weight matrix for each will be `(6x6)` which is calculated with `d_in x d_out`

   ```python
   # Note these are all random numbers
   Wq = tensor([[0.4993, 0.2528, 0.9087, 0.4387, 0.0761, 0.6710],
        [0.4096, 0.0766, 0.3847, 0.5818, 0.5646, 0.5658],
        [0.2798, 0.7151, 0.1058, 0.6133, 0.2456, 0.3947],
        [0.5394, 0.9090, 0.6069, 0.5502, 0.2611, 0.0625],
        [0.9705, 0.0770, 0.5557, 0.5505, 0.4124, 0.5342],
        [0.1674, 0.8521, 0.7174, 0.2237, 0.0440, 0.4619]], dtype=torch.float64)

   Wk = tensor([[0.2858, 0.4111, 0.6800, 0.7871, 0.7942, 0.9427],
        [0.2936, 0.8079, 0.2960, 0.4484, 0.4080, 0.0224],
        [0.6710, 0.9872, 0.1064, 0.0914, 0.7141, 0.2901],
        [0.4991, 0.7919, 0.2702, 0.5881, 0.2960, 0.7207],
        [0.2867, 0.9855, 0.7803, 0.1783, 0.6278, 0.9921],
        [0.7149, 0.9552, 0.8109, 0.9274, 0.8857, 0.5978]], dtype=torch.float64)

   Wv = tensor([[0.0403, 0.9850, 0.9636, 0.2161, 0.2303, 0.7819],
        [0.6405, 0.3427, 0.0892, 0.8515, 0.8595, 0.4642],
        [0.8738, 0.0162, 0.1105, 0.0801, 0.5262, 0.2379],
        [0.0604, 0.5162, 0.9510, 0.8857, 0.4536, 0.1404],
        [0.3554, 0.6543, 0.0071, 0.3822, 0.7250, 0.8788],
        [0.4789, 0.0458, 0.6999, 0.9771, 0.9141, 0.1221]], dtype=torch.float64)
   ```

4. **Step 4:** Calculate Quaries, Keys, Values matrix.

   - We use the formula `(Input * Wq, Input * Wk, Input * Wv) -> (1x3x6)`
   - breaking it down input dim is `1x3x6` and dim for each weight matrix is `6x6`

   ```python
   # Note these are all random numbers
   quaries = tensor([[[0.9763, 0.3485, 0.3802, 0.3914, 0.2063, 0.8732],
         [0.7284, 0.0698, 0.4273, 0.5182, 0.3345, 0.5293],
         [0.0566, 0.5583, 0.0058, 0.6483, 0.8398, 0.5502]]],
       dtype=torch.float64)

   keys = tensor([[[0.0429, 0.5595, 0.3988, 0.1734, 0.0249, 0.3626],
         [0.5859, 0.9974, 0.0444, 0.3262, 0.6001, 0.3775],
         [0.8944, 0.9160, 0.1576, 0.2976, 0.8728, 0.2907]]],
       dtype=torch.float64)

   values = tensor([[[0.4535, 0.7414, 0.1504, 0.6988, 0.6502, 0.8611],
         [0.5290, 0.1509, 0.2134, 0.8503, 0.3689, 0.2023],
         [0.1310, 0.2897, 0.6613, 0.5790, 0.4172, 0.6829]]],
       dtype=torch.float64)
   ```

5. **Step 5:** Unroll last dimension of `quaries`, `keys` and `values` to include `num_heads` and `head_dim`

   - `head_dim = d_out / num_heads => 6/2 = 3`
   - so we have the below

   ```plaintext
    (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
    (1, 3, 6) -> (1, 3, 2, 3)
   ```

   ```python
   # Note these are all random numbers

   # Reshaped quaries
   tensor([[[[0.8767, 0.2653, 0.2036],
          [0.8964, 0.0012, 0.1222]],

         [[0.8092, 0.0758, 0.8391],
          [0.7259, 0.9211, 0.7165]],

         [[0.4710, 0.6952, 0.8075],
          [0.8630, 0.5784, 0.2735]]]], dtype=torch.float64)

   # Reshaped keys
   tensor([[[[0.5248, 0.3205, 0.2382],
          [0.0070, 0.7636, 0.3844]],

         [[0.9814, 0.8952, 0.4943],
          [0.5488, 0.1439, 0.4917]],

         [[0.6674, 0.5351, 0.6264],
          [0.9059, 0.3096, 0.5012]]]], dtype=torch.float64)

   # Reshaped values
   tensor([[[[0.8743, 0.3415, 0.7396],
          [0.6737, 0.4974, 0.9672]],

         [[0.5274, 0.0807, 0.9981],
          [0.4811, 0.3248, 0.2939]],

         [[0.1379, 0.6857, 0.7784],
          [0.5672, 0.6477, 0.1868]]]], dtype=torch.float64)
   ```

6. **Step 6:** Group matrics by `number of heads`

   - this is done so that when performing matrix, the heads can be matched to each other from each weight matrix

   ```plaintext
    (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
    (1, 3, 2, 3) -> (1, 2, 3, 3)
   ```

   ```python
    # Note these are all random numbers

    # transposed quaries matrix
    tensor([[[[0.0299, 0.7057, 0.1425],  # HEAD 1, TOKEN 1
             [0.5029, 0.6294, 0.3265],     # HEAD 1, TOKEN 2
             [0.8386, 0.7803, 0.8877]],    # HEAD 1, TOKEN 3

            [[0.0808, 0.7281, 0.7343],     # HEAD 2, TOKEN 1
             [0.5948, 0.8757, 0.6526],     # HEAD 2, TOKEN 2
             [0.8280, 0.1269, 0.9827]]]],  # HEAD 2, TOKEN 3
          dtype=torch.float64)

    # transposed keys matrix
    tensor([[[[0.2260, 0.7611, 0.6772],  # HEAD 1, TOKEN 1
             [0.7446, 0.4209, 0.6467],     # HEAD 1, TOKEN 2
             [0.4521, 0.1769, 0.1615]],    # HEAD 1, TOKEN 3

            [[0.6787, 0.7103, 0.8188],     # HEAD 2, TOKEN 1
             [0.5338, 0.8099, 0.9866],     # HEAD 2, TOKEN 2
             [0.0227, 0.4601, 0.4753]]]],  # HEAD 2, TOKEN 3
          dtype=torch.float64)

    # transposed values matrix
    tensor([[[[0.8375, 0.7430, 0.8563],  # HEAD 1, TOKEN 1
             [0.1214, 0.1560, 0.0729],     # HEAD 1, TOKEN 2
             [0.1887, 0.9912, 0.1640]],    # HEAD 1, TOKEN 3

            [[0.7458, 0.6515, 0.1220],     # HEAD 2, TOKEN 1
             [0.6801, 0.4366, 0.6720],     # HEAD 2, TOKEN 2
             [0.8135, 0.6831, 0.7280]]]],  # HEAD 2, TOKEN 3
          dtype=torch.float64)
   ```

7. **Step 7:** Find attention scores

   - `quaries * keys.T(2,3)`
   - so we're once again swapping the position of `head_dim` and `num_tokens`

   ```plaintext
    (b, num_heads, num_tokens, head_dim) * (b, num_heads, head_dim, num_tokens) ->(b, num_heads, num_tokens, num_tokens)
    (1, 2, 3, 3) * (1, 2, 3, 3) -> (1, 2, 3, 3)
   ```

   ```python
   # Note these are all random numbers

   # quaries matrix
   # (b, num_heads, num_tokens, head_dim)
   tensor([[[[0.0299, 0.7057, 0.1425],  # HEAD 1
          [0.5029, 0.6294, 0.3265],     # HEAD 1
          [0.8386, 0.7803, 0.8877]],    # HEAD 1

         [[0.0808, 0.7281, 0.7343],     # HEAD 2
          [0.5948, 0.8757, 0.6526],     # HEAD 2
          [0.8280, 0.1269, 0.9827]]]],  # HEAD 2
        dtype=torch.float64)

   # keys.transposed(2,3)
   # (b, num_heads, head_dim, num_tokens)
   tensor([[[[0.2260, 0.7446, 0.4521],  # HEAD 1
          [0.7611, 0.4209, 0.1769],     # HEAD 1
          [0.6772, 0.6467, 0.1615]],    # HEAD 1

         [[0.6787, 0.5338, 0.0227],     # HEAD 2
          [0.7103, 0.8099, 0.4601],     # HEAD 2
          [0.8188, 0.9866, 0.4753]]]],  # HEAD 2
        dtype=torch.float64)
    #

   # attention scores matrix = quaries * keys.transpose(2,3)
   # (b, num_heads, num_tokens, num_tokens)
    tensor([[[[0.6404, 0.4114, 0.1614], # attention scores for a token on other tokens
          [0.8138, 0.8505, 0.3914],
          [1.3846, 1.5269, 0.6605]],

         [[1.1733, 1.3573, 0.6858],
          [1.5600, 1.6706, 0.7266],
          [1.4567, 1.5143, 0.5443]]]], dtype=torch.float64)
   ```

   The reason the resulting attention scores have shape **`(b, num_heads, num_tokens, num_tokens)`** is because of how matrix multiplication works on the **last two dimensions** of the `queries` and `keys.transpose(2, 3)` tensors.

   - The last two dimensions of `queries` are **`(num_tokens, head_dim)`**.
   - The last two dimensions of `keys.transpose(2, 3)` are **`(head_dim, num_tokens)`**.

   When we multiply these two, the **inner dimensions** (head_dim) must match, and the **outer dimensions** become the shape of the resulting matrix:

   ```plaintext
   (num_tokens, head_dim) × (head_dim, num_tokens) = (num_tokens, num_tokens)
   ```

   This gives us a **token-to-token similarity matrix** for each head — essentially, it measures how much each token (as a query) attends to every other token (as a key).

   The transpose on the keys is necessary precisely to align the dimensions this way, so that the dot product can be computed between token embeddings and yield a square attention score matrix over the sequence length.

8. **Step 8:** Finding attention weights

   - Max attention scores to implement causal attention

   ```python
   tensor([[[[0.6404, -inf, -inf], # attention scores for a token on other tokens
           [0.8138, 0.8505, -inf],
           [1.3846, 1.5269, 0.6605]],

           [[1.1733, -inf, -inf],
           [1.5600, 1.6706, -inf],
           [1.4567, 1.5143, 0.5443]]]], dtype=torch.float64)
   ```

   - Divid by `sqrt(head_dim) = sqrt(d_out/num_heads) => sqrt(6/2)=sqrt(3)`

   ```python
   tensor([[[[0.3697,   -inf,   -inf],  # HEAD 1
          [0.4698, 0.4910,   -inf],
          [0.7994, 0.8816, 0.3813]],

         [[0.6774,   -inf,   -inf],     # HEAD 1
          [0.9007, 0.9645,   -inf],
          [0.8410, 0.8743, 0.3143]]]], dtype=torch.float64)
   ```

   - Apply softmax to calucate attention weights

   ```python
   tensor([[[[1.0000, 0.0000, 0.0000],  # HEAD 1
          [0.4947, 0.5053, 0.0000],
          [0.3644, 0.3956, 0.2399]],

         [[1.0000, 0.0000, 0.0000],     # HEAD 2
          [0.4840, 0.5160, 0.0000],
          [0.3811, 0.3939, 0.2250]]]])
   ```

   - We can also implement dropout(THIS IS STEP IS OPTIONAL THOUGH. BUT IT'S CRUCIAL AS WELL)

   ```python
   tensor([[[[0.0000, 0.0000, 0.0000],  # HEAD 1, TOKEN 1
          [0.9894, 0.0000, 0.0000],     # HEAD 1, TOKEN 2
          [0.7289, 0.0000, 0.4798]],    # HEAD 1, TOKEN 3

         [[0.0000, 0.0000, 0.0000],     # HEAD 2, TOKEN 1
          [0.0000, 1.0319, 0.0000],     # HEAD 2, TOKEN 2
          [0.7621, 0.7879, 0.0000]]]])  # HEAD 2, TOKEN 3
   ```

9. **Step 9:** calculate the context vector

   - `context vector = attention weights(b, num_heads, num_tokens, num_tokens) * values(b, num_head, num_tokens, head_dim)`

   ```python

   # attention weights
   tensor([[[[1.0000, 0.0000, 0.0000],  # HEAD 1
          [0.4947, 0.5053, 0.0000],
          [0.3644, 0.3956, 0.2399]],

         [[1.0000, 0.0000, 0.0000],     # HEAD 2
          [0.4840, 0.5160, 0.0000],
          [0.3811, 0.3939, 0.2250]]]])


   # values weight matrix
   tensor([[[[0.8375, 0.7430, 0.8563],  # HEAD 1
          [0.1214, 0.1560, 0.0729],     # HEAD 1
          [0.1887, 0.9912, 0.1640]],    # HEAD 1

         [[0.7458, 0.6515, 0.1220],     # HEAD 2
          [0.6801, 0.4366, 0.6720],     # HEAD 2
          [0.8135, 0.6831, 0.7280]]]],  # HEAD 2
        dtype=torch.float64)


   # context vector = attention weight * values
   tensor([[[[0.8375, 0.7430, 0.8563],  # HEAD 1
          [0.4757, 0.4464, 0.4605],
          [0.3985, 0.5703, 0.3803]],

         [[0.7458, 0.6515, 0.1220],     # HEAD 2
          [0.7119, 0.5406, 0.4058],
          [0.7352, 0.5740, 0.4750]]]], dtype=torch.float64)
   ```

10. **Step 10:** Reformat context vectors.

    ```plaintext
    (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
    (1, 2, 3, 3) -> (1, 3, 2, 3)
    ```

    ```python
    tensor([[[[0.8375, 0.7430, 0.8563], # TOKEN 1, HEAD 1
            [0.7458, 0.6515, 0.1220]],  # TOKEN 1, HEAD 2

            [[0.4757, 0.4464, 0.4605],  # TOKEN 2, HEAD 1
            [0.7119, 0.5406, 0.4058]],  # TOKEN 2, HEAD 2

            [[0.3985, 0.5703, 0.3803],  # TOKEN 3, HEAD 1
            [0.7352, 0.5740, 0.4750]]]],# TOKEN 2, HEAD 2
            dtype=torch.float64)
    ```

    - this is to regroup the context back to num_tokens available. so this we see there are 3 tokens with 2 head and their respective attention heads

11. Step 11: Combine result from multiple attention heads

    - this is done by flattening the context vector to form `(1, 3, 6)` shape which is the same shape the original input vector came with

    ```plaintext
    so the idea is to go from

    (b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
    (1, 2, 3, 3) -> (1, 3, 6)
    ```

    ```python
    tensor([[[0.8375, 0.7430, 0.8563, 0.4757, 0.4464, 0.4605],  # TOKEN 1
         [0.3985, 0.5703, 0.3803, 0.7458, 0.6515, 0.1220],      # TOKEN 2
         [0.7119, 0.5406, 0.4058, 0.7352, 0.5740, 0.4750]]],    # TOKEN 3
       dtype=torch.float64)
    ```
