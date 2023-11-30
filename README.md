# CHAT-GPT--2-small

## Task 1 | GPT-2 Model & Checkpoints

The project outlined involves implementing a scaled-down version of the GPT-2 small model using Python and the PyTorch library. The goal is to build a transformer-based language model similar to GPT-2, which has 125 million parameters. Here is a description of the project, including its main components:

### Project Description

**Objective:**
Develop a custom GPT-2 small model from scratch, closely following the original design but without utilizing any pre-existing transformer libraries. The implementation will be done in Python using PyTorch.

**Key Components:**

1. **Token and Positional Embeddings:**
   - Implement token embeddings to convert token IDs to vectors.
   - Implement positional encodings to inject position information into the input embeddings, ensuring the model can respect the order of tokens.

2. **Multi-Head Self-Attention:**
   - Design a self-attention mechanism that allows the model to weigh the importance of different tokens within the input sequence differently.
   - Implement the attention mechanism across multiple 'heads', enabling the model to attend to different parts of the sequence simultaneously in parallel processing streams.

3. **Transformer Layers:**
   - Stack multiple transformer layers, each consisting of a multi-head self-attention mechanism followed by a point-wise feed-forward network.
   - Include residual connections and layer normalization around each sub-component (self-attention and feed-forward network) within the transformer layers.

4. **Feed-Forward Networks:**
   - Develop a feed-forward network comprising two linear transformations with a ReLU activation in the middle, applied identically across all positions.

5. **Model Architecture:**
   - Assemble the token embeddings, positional encodings, and transformer layers into a coherent model architecture that follows the specifications of the GPT-2 small variant.

**Testing and Validation:**
- Test the model to ensure it can handle input sequences and produce output.
- Load pre-trained GPT-2 125M model checkpoints (if available) to compare the performance of the custom implementation against the original model's outputs on the same inputs.

**Deliverables:**
- A complete Python codebase featuring the custom GPT-2 implementation.
- Documentation and comments within the code explaining each component and its role in the model.
- A testing suite or demonstration script that inputs a sample sequence into the model and outputs a prediction, showcasing the model's language generation capabilities.

**Resources:**
- The original GPT-2 paper, for architectural insights and technical specifications.
- Online tutorials and series, such as Andrej Karpathy's "makemore" series, for practical coding examples and additional context on transformer models.

**Challenges:**
- Ensuring the model reaches a comparable level of complexity with 125 million parameters, without direct guidance from pre-built libraries.
- Debugging and validating each part of the model to align with the expected behavior of GPT-2.
- Optimizing the model for efficient training and inference, given the large number of parameters.

**Outcome:**
Upon completion, the project will yield a functioning language model based on the GPT-2 architecture capable of generating text and performing other language-based tasks, serving as a foundational example for further research and development in transformer models.

## CODE

```python
import torch
import torch.nn as nn
import math

# Configuration for a small GPT-2 model
class GPT2Config:
    vocab_size = 50257
    max_position_embeddings = 1024
    n_layers = 12
    n_heads = 12
    n_embd = 768
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02

config = GPT2Config()

# Define the scaled dot product attention function
def scaled_dot_product_attention(query, key, value):
    temp = query.bmm(key.transpose(1, 2)) / math.sqrt(query.size(-1))
    softmax = nn.Softmax(dim=-1)
    return softmax(temp).bmm(value)

# Define a single head for the Multi-Head Attention
class AttentionHead(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.query = nn.Linear(embd_dim, embd_dim)
        self.key = nn.Linear(embd_dim, embd_dim)
        self.value = nn.Linear(embd_dim, embd_dim)

    def forward(self, hidden_state):
        return scaled_dot_product_attention(
            self.query(hidden_state), self.key(hidden_state), self.value(hidden_state)
        )

# Define the Multi-Head Attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embd_dim, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(embd_dim) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads * embd_dim, embd_dim)

    def forward(self, hidden_state):
        attention = [head(hidden_state) for head in self.heads]
        concatenated = torch.cat(attention, dim=-1)
        return self.linear(concatenated)

# Define the Pointwise Feed Forward layer
class PointwiseFeedForward(nn.Module):
    def __init__(self, embd_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embd_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embd_dim)

    def forward(self, hidden_state):
        return self.linear2(nn.functional.relu(self.linear1(hidden_state)))

# Define a single Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embd_dim, n_heads, ff_dim, layer_norm_epsilon):
        super().__init__()
        self.attention = MultiHeadAttention(embd_dim, n_heads)
        self.feed_forward = PointwiseFeedForward(embd_dim, ff_dim)
        self.layer_norm1 = nn.LayerNorm(embd_dim, eps=layer_norm_epsilon)
        self.layer_norm2 = nn.LayerNorm(embd_dim, eps=layer_norm_epsilon)

    def forward(self, hidden_state):
        attention_output = self.attention(hidden_state)
        norm1 = self.layer_norm1(hidden_state + attention_output)
        feed_forward_output = self.feed_forward(norm1)
        norm2 = self.layer_norm2(norm1 + feed_forward_output)
        return norm2

# Define the full GPT-2 model
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embd_dim = config.n_embd
        self.token_embedding = nn.Embedding(config.vocab_size, self.embd_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, self.embd_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.embd_dim, config.n_heads, 4 * self.embd_dim, config.layer_norm_epsilon) for _ in range(config.n_layers)]
        )
        self.layer_norm = nn.LayerNorm(self.embd_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, positions_ids=None):
        if positions_ids is None:
            positions_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        tokens = self.token_embedding(input_ids)
        positions = self.position_embedding(positions_ids)
        
        x = tokens + positions

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)
        return x

# Example usage
model = GPT2(config)
input_ids = torch.randint(0, config.vocab_size,

 (1, 1024))  # Random input for demonstration
output = model(input_ids)
print(output)
```

This code is a basic scaffold and does not include functionality such as the attention mask (which is crucial for handling variable-length sequences), the output layer that converts transformer outputs to token logits, nor does it handle loading pre-trained weights or any of the necessary components for training such as a loss function or optimizer.


## Task 2 | Transformer Architectural Changes

Implementing the full GPT-2 model with the suggested alterations (Rotary Positional Embedding, Group Query Attention, and Sliding Window Attention) 

### Rotary Positional Embedding
Rotary Positional Embeddings (RoPE) encode relative position information directly into the attention mechanism, which could potentially capture more nuanced dependencies between tokens.

```python
import torch

def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: t.repeat_interleave(2, dim=-1), sincos)
    return (x * cos) + (torch.roll(x, shifts=1, dims=-1) * sin)
```

To integrate RoPE into the GPT-2 model, you would replace the existing positional embeddings with these rotary embeddings within the attention calculations.

### Group Query Attention
Group Query Attention would involve grouping queries together and performing attention with these grouped queries, which could improve the model's multi-task learning capabilities.

```python
def group_query_attention(query, key, value, num_groups):
    # Split queries into groups
    group_size = query.size(2) // num_groups
    query_groups = query.view(*query.size()[:2], num_groups, group_size)
    
    # Perform attention within each group
    attention_output = []
    for i in range(num_groups):
        group_attn_output = scaled_dot_product_attention(query_groups[:,:,i,:], key, value)
        attention_output.append(group_attn_output)
    
    # Concatenate the outputs of each group
    return torch.cat(attention_output, dim=-1)
```

### Sliding Window Attention
The Sliding Window Attention mechanism is particularly useful for processing longer sequences by limiting the attention computation to a fixed-size window around each token.

```python
def sliding_window_attention(query, key, value, window_size):
    # Assume query, key, and value are all the same size for simplicity
    batch_size, seq_length, dim = query.size()
    attention_scores = torch.empty((batch_size, seq_length, window_size), device=query.device)
    
    # Compute attention scores for a sliding window
    for i in range(seq_length):
        start = max(0, i - window_size // 2)
        end = min(seq_length, i + window_size // 2 + 1)
        attention_scores[:, i, :end-start] = torch.bmm(query[:, i:i+1, :], key[:, start:end, :].transpose(1, 2))
    
    # Apply softmax to get attention probabilities
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    
    # Compute weighted sum to get the attention output
    attention_output = torch.bmm(attention_probs, value[:, start:end, :])
    return attention_output
```

### Full Code Overview
The full code would involve defining a PyTorch `nn.Module` for the GPT-2 model, then incorporating these functions into the self-attention computations of the model. This would require modifying the `forward` methods of the attention and transformer block modules to use the new attention mechanisms.

### Explanation
Each function provided here is a simplified version of the component they represent. The `apply_rotary_pos_emb` function applies rotary embeddings to the input sequence. The `group_query_attention` function modifies the attention mechanism to operate on grouped queries. The `sliding_window_attention` function computes attention scores within a local window for each token.

## Task 3: Training Loop Implementation
Creating a training loop for a deep learning model involves several key steps: preparing the dataset, defining the model, specifying the loss function and optimizer, and then iterating over the dataset in epochs and batches. Below, I'll outline pseudocode for each of the three specified training setups: Single GPU, Distributed Data Parallel (DDP), and Fully Sharded Data Parallel (FSDP). Note that actual implementation would require a full Python script and environment setup with appropriate data, which is not feasible here.

### 1. Single GPU Training Loop
For training on a single GPU, you can set up a simple loop that processes your data in batches, computes the loss, and updates the model parameters.

```python
import torch

# Assuming model, dataset, optimizer, and loss function are defined

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 2. Distributed Data Parallel (DDP)
To extend the training loop to multiple GPUs using DDP, you can wrap your model with `torch.nn.parallel.DistributedDataParallel`. You'll also need to set up the distributed environment and adjust your data loading accordingly.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed

# Setup (before the loop)
dist.init_process_group(backend='nccl')
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# Adjust dataloader for distributed training
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, ...)

# Training loop (similar to single GPU, but with sampler.set_epoch at the beginning of each epoch)
```

### 3. Fully Sharded Data Parallel (FSDP)
Implementing a complete training loop with Single GPU, Distributed Data Parallel (DDP), and Fully Sharded Data Parallel (FSDP)

### Single GPU Training Loop

For a single GPU, you would follow a standard training loop in PyTorch.

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Sample model, replace with actual model
model = nn.Linear(10, 2)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
data_loader = DataLoader(your_dataset, batch_size=32, shuffle=True)

# Move model to GPU
device = torch.device("cuda:0")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Distributed Data Parallel (DDP)

For DDP, the setup is more involved. You need to initialize the process group and wrap your model in `DistributedDataParallel`.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = model.to(device)
ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# Distributed Sampler
sampler = DistributedSampler(dataset)

# Update data_loader to use DistributedSampler
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=sampler)

# Training loop
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for inputs, labels in data_loader:
        # ...same as single GPU training loop
```

### Fully Sharded Data Parallel (FSDP)

FSDP is a more advanced technique that shards the model parameters across all processes to significantly reduce memory overhead. This requires PyTorch version 1.10 or newer.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)

# Training loop is similar to DDP, but the model is now an instance of FSDP
```

### Full Detailed Code Explanation

In a full implementation:

1. **Setup**: You would include code to parse arguments or read from a configuration file to determine which mode to run (single GPU, DDP, FSDP).
2. **Data Loading**: Your dataset should be properly formatted and preprocessed for use with a `DataLoader`.
3. **Model Definition**: You need a concrete model definition that you wish to train. The sample linear model is just a placeholder.
4. **DDP Initialization**: For DDP, each process should know its rank and the world size (total number of processes). This is often done by parsing command-line arguments.
5. **FSDP Initialization**: FSDP requires additional setup, like gradient clipping and memory management strategies, for full functionality.
6. **Training Loop**: The actual training loop doesn't change much across different modes, but the setup and teardown do.
7. **Evaluation**: After each epoch, you'd evaluate the model on a validation set, which is not shown here.
8. **Checkpoints**: You'd save checkpoints after certain intervals or conditions.
9. **Logging**: Throughout the training process, you'd log necessary metrics for monitoring.
