
import torch
import torch.nn as nn


class selfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int) -> None:
        super(selfAttention, self).__init__()

        self.embed_size: int = embed_size
        self.heads: int = heads
        self.head_dim: int = embed_size // heads

        assert (self.head_dim * heads == embed_size), "embed_size must be divisible by heads"

        # Project the complete embedding before splitting it into attention heads
        self.values: nn.Linear = nn.Linear(embed_size, embed_size, bias=False)
        self.keys: nn.Linear = nn.Linear(embed_size, embed_size, bias=False)
        self.queries: nn.Linear = nn.Linear(embed_size, embed_size, bias=False)

        self.fully_connected_out: nn.Linear = nn.Linear(
            heads * self.head_dim, embed_size, bias=False
        )

    #forward propagation
    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        masks: torch.Tensor | None = None
    ) -> torch.Tensor:

        N: int = query.shape[0]  #batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  #sequence length

        # Apply independent learned projections to complete embeddings
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        #split embedding into heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Rearrange Query and Key dimensions for matrix multiplication
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)

        # Q × Kᵀ = Attention scores
        score: torch.Tensor = torch.matmul(
            queries, keys.transpose(-2, -1)
        )

        # Scale attention scores before applying the mask
        score = score / (self.head_dim ** 0.5)

        # Set masked attention scores to negative infinity
        if masks is not None:
            # Convert the attention mask to boolean values
            masks = masks.to(dtype=torch.bool)

            # Use the minimum finite value to avoid NaN in fully masked rows
            score = score.masked_fill(
                ~masks,
                torch.finfo(score.dtype).min
            )

        # Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k)) @ V
        attention: torch.Tensor = torch.softmax(
            score, dim=3
        )

        if masks is not None:
            # Force masked positions to zero, including fully masked rows
            attention = attention.masked_fill(~masks, 0.0)

        multihead_out: torch.Tensor = (
            torch.einsum("nhql,nlhd -> nqhd", [attention, values])
            .reshape(N, query_len, self.heads * self.head_dim)
        )
        #attention shape = (N, heads, query_len, key_len)
        #values shape = (N, value_len, heads, heads_dim)
        #after einsum = (N, query_len, heads, heads_dim)

        multihead_out = self.fully_connected_out(multihead_out)

        return multihead_out


class transformerBlock(nn.Module):
    def __init__(
        self,
        embed_size: int,
        heads: int,
        dropout: float,
        forward_expansion: int
    ) -> None:
        super(transformerBlock, self).__init__()

        self.attention: selfAttention = selfAttention(embed_size, heads)
        self.norm1: nn.LayerNorm = nn.LayerNorm(embed_size)
        self.norm2: nn.LayerNorm = nn.LayerNorm(embed_size)

        self.feed_forward: nn.Sequential = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        masks: torch.Tensor | None = None
    ) -> torch.Tensor:

        attention: torch.Tensor = self.attention(values, keys, query, masks)  # Compute multi-head attention

        # Apply dropout before residual addition and LayerNorm (Post-LN)
        x: torch.Tensor = self.norm1(
            query + self.dropout(attention)
        )

        forward: torch.Tensor = self.feed_forward(x)  # Transform features through the feed-forward network

        # Apply dropout before the second residual addition and LayerNorm
        out: torch.Tensor = self.norm2(
            x + self.dropout(forward)
        )

        return out  # Return the transformed output


class Encoder(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()

        self.embed_size: int = embed_size

        # Initialize token and positional embeddings
        self.word_embeddings: nn.Embedding = nn.Embedding(
            source_vocab_size, embed_size
        )
        self.position_embeddings: nn.Embedding = nn.Embedding(
            max_length, embed_size
        )

        # Initialize independent Transformer encoder blocks
        self.layers: nn.ModuleList = nn.ModuleList([
            transformerBlock(
                embed_size=embed_size,
                heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion
            )
            for _ in range(num_layers)
        ])

        # Initialize embedding dropout
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None
    ) -> torch.Tensor:

        N, seq_len = x.shape  # Extract batch size and sequence length

        # Generate positional indices
        positions: torch.Tensor = torch.arange(
            seq_len, device=x.device
        ).unsqueeze(0).expand(N, seq_len)

        # Combine token and positional embeddings
        out: torch.Tensor = self.dropout(
            self.word_embeddings(x) +
            self.position_embeddings(positions)
        )

        # Apply Transformer encoder blocks sequentially
        for layer in self.layers:
            out = layer(out, out, out, masks)

        return out  # Return contextual token representations


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        device: str | torch.device,
    ) -> None:
        super(DecoderBlock, self).__init__()

        self.attention: selfAttention = selfAttention(embed_size, heads)
        self.norm: nn.LayerNorm = nn.LayerNorm(embed_size)

        self.TransformerBlock: transformerBlock = transformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion
        )

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        value: torch.Tensor,
        key: torch.Tensor,
        source_mask: torch.Tensor | None,
        target_mask: torch.Tensor | None
    ) -> torch.Tensor:

        attention: torch.Tensor = self.attention(
            x, x, x, target_mask
        )

        # Apply masked self-attention residual connection using Post-LN
        query: torch.Tensor = self.norm(
            x + self.dropout(attention)
        )

        # Apply encoder-decoder cross-attention and feed-forward network
        out: torch.Tensor = self.TransformerBlock(
            value, key, query, source_mask
        )

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        device: str | torch.device,
        max_length: int,
    ) -> None:
        super(Decoder, self).__init__()

        self.device: str | torch.device = device

        self.word_embeddings: nn.Embedding = nn.Embedding(
            target_vocab_size, embed_size
        )

        # Initialize positional embeddings for target token positions
        self.position_embeddings: nn.Embedding = nn.Embedding(
            max_length, embed_size
        )

        self.layers: nn.ModuleList = nn.ModuleList([
            DecoderBlock(
                embed_size,
                heads,
                forward_expansion,
                dropout,
                device,
            )
            for _ in range(num_layers)
        ])

        self.fully_connected_out: nn.Linear = nn.Linear(
            embed_size, target_vocab_size
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        source_mask: torch.Tensor | None,
        target_mask: torch.Tensor | None
    ) -> torch.Tensor:

        N, seq_len = x.shape

        positions: torch.Tensor = torch.arange(
            0, seq_len, device=x.device
        ).unsqueeze(0).expand(N, seq_len)

        x = self.dropout(
            self.word_embeddings(x) +
            self.position_embeddings(positions)
        )

        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                encoder_out,
                source_mask,
                target_mask
            )

        # Project decoder hidden states into target vocabulary logits
        out: torch.Tensor = self.fully_connected_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        source_pad_idx: int,
        target_pad_idx: int,
        embed_size: int = 512,
        num_layers: int = 6,
        forward_expansion: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        device: str | torch.device = "cpu",
        max_length: int = 100,
    ) -> None:
        super().__init__()

        self.source_pad_idx: int = source_pad_idx
        self.target_pad_idx: int = target_pad_idx

        self.encoder: Encoder = Encoder(
            source_vocab_size=source_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length,
        )

        self.decoder: Decoder = Decoder(
            target_vocab_size=target_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length,
        )

    def make_source_mask(
        self,
        source: torch.Tensor
    ) -> torch.Tensor:

        return (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_target_mask(
        self,
        target: torch.Tensor
    ) -> torch.Tensor:

        N, target_len = target.shape

        causal_mask: torch.Tensor = torch.tril(
            torch.ones(
                target_len,
                target_len,
                device=target.device,
                dtype=torch.bool
            )
        ).unsqueeze(0).unsqueeze(0)

        padding_mask: torch.Tensor = (
            target != self.target_pad_idx
        ).unsqueeze(1).unsqueeze(2)

        return causal_mask & padding_mask

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        source_mask: torch.Tensor = self.make_source_mask(source)
        target_mask: torch.Tensor = self.make_target_mask(target)

        encoder_out: torch.Tensor = self.encoder(source, source_mask)

        output: torch.Tensor = self.decoder(
            target,
            encoder_out,
            source_mask,
            target_mask
        )

        return output
