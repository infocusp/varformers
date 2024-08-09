import math

import torch
import torch.utils.checkpoint
from torch import nn

class LongformerConfig():
    def __init__(self, hidden_size = 128,
                 num_attention_heads = 8,
                 attention_probs_dropout_prob=0.1,
                 attention_window: tuple[list, int] = None, 
                 attention_dilation: tuple[list, int] = None,                 
                 autoregressive: bool = False, 
                 attention_mode: str = 'sliding_chunks', **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2', 'sliding_chunks_no_overlap']

class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        # separate projection layers for tokens with global attention
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

        self.config = config
        
    def _pad_to_window_size(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_window: int,
        pad_token_id: int = 1,
        factor = 2
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        
        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"

        batch_size, seq_len, embed_dim = inputs_embeds.shape

        padding_len = (factor*attention_window - seq_len % (factor*attention_window)) % (factor*attention_window)

        # this path should be recorded in the ONNX export, it is fine with padding_len == 0 as well
        if padding_len > 0:  

            inputs_embeds_padding = inputs_embeds.new_full(
                (batch_size, padding_len, embed_dim),
                pad_token_id
            )   
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = nn.functional.pad(
                attention_mask.squeeze(dim=-1), (0, padding_len), value=True
            ).unsqueeze(dim=-1)   # no attention on the padding tokens
            
        return inputs_embeds, attention_mask, padding_len    

    def forward(
        self,
        hidden_states,
        attention_mask=None,       
        is_global_attn=True,
        output_attentions=False,
    ):
        
        hidden_states, attention_mask, padding_len  = self._pad_to_window_size(hidden_states, 
                                                                               attention_mask,
                                                                               2*self.one_sided_attn_window_size, 
                                                                               factor=1)
        
        # is_index_masked is same as attention mask
        is_index_masked = attention_mask
        
        # create a global attention mask to have just CLS token in it
        is_index_global_attn = attention_mask.new_empty(attention_mask.size())
        is_index_global_attn[:, 0, :] = True
        is_index_global_attn[:, 1:, :] = False      


        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        batch_size, seq_len, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)
        
        # mask the key_vectors (padding) with 0
        key_vectors = key_vectors.masked_fill(attention_mask, 0)

        query_vectors = query_vectors.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_vectors = key_vectors.view(batch_size, seq_len, self.num_heads, self.head_dim)    
        
        # calculate partial local attention
        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], (
            f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads},"
            f" {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
        )
        
        if is_global_attn:
            
            # get global indices
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            
            # (A): calculate (partial) attention where each token (from query) attends global token (from key)
            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
            
            # concat (A) and partial local attention            
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1) # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)

            del global_key_attn_scores # free memory
        
        # fill the padding with -inf to get ignored from softmax    
        attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(dim=-1), -float("inf"))
        
        # mask the zero values to -inf in the last valid self.one_sided_attn_window_size tokens before padding 
        attn_scores = attn_scores.masked_fill(attn_scores==0, -float("inf"))

        attn_probs = nn.functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # softmax inserts NaN if all positions are masked, replace them with 0
        attn_probs = attn_probs.masked_fill(attention_mask.unsqueeze(dim=-1), 0)
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores

        # apply dropout
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        value_vectors = value_vectors.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        if is_global_attn:
            
            # (B) calculate complete attention where each token (from (A)) attends global token (from value)
            # calculate complete local attention
            # (X): Add ((B) + complete local attention)
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # calculate complete local attention only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim).contiguous()

        # (Y) calculate attention where global token (from global_query) attends all other tokens from global_key and global_value
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # replace the attention values in (X) at global indices with (Y)
            attn_output[is_index_global_attn_nonzero] = global_attn_output[is_local_index_global_attn_nonzero]
            
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0
        
        outputs = attn_output[:,:attn_output.shape[1]-padding_len,:] # unpad the attention output

        return (outputs, attn_probs, global_attn_probs) if (is_global_attn and output_attentions) else outputs

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = nn.functional.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):

        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap, onnx_export: bool = False):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        if not onnx_export:
            # non-overlapping chunks of size = 2w
            hidden_states = hidden_states.view(
                hidden_states.size(0),
                torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
                window_overlap * 2,
                hidden_states.size(2),
            )
            
            # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1

            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)


        chunk_size = [
            hidden_states.size(0),
            torch.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1,
            window_overlap * 2,
            hidden_states.size(2),
        ]

        overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[
                :, chunk * window_overlap : chunk * window_overlap + 2 * window_overlap, :
            ]
        return overlapping_chunks

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
            ending_input, -float("inf")
        ).where(ending_mask.bool(), ending_input)

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w  with an
        overlap of size w
        """
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
            seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        query = self._chunk(query, window_overlap, False)
        key = self._chunk(key, window_overlap, False)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower 
        # triangles (attention from a word to window_overlap previous words). The following column is attention 
        # score from each word to itself, then followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        # mask invalid locations for example first window_overlap values before first word or window_overlap values after last word
        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = value.size()

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads,
            torch.div(seq_len, window_overlap, rounding_mode="trunc"),
            window_overlap,
            2 * window_overlap + 1,
        )

        # group batch_size and num_heads dimensions into one
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # helper variable
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero[:-1],
            is_local_index_global_attn_nonzero[:-1],
        ) # slicing [:-1] as we do not need the last tensor

    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # create only global key vectors
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query_vectors, key_vectors_only_global))

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        batch_size = attn_probs.shape[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        
        # get value vectors for global only
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]

        # use `matmul` because `einsum` crashes sometimes with fp16
        # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2), value_vectors_only_global.transpose(1, 2)
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        hidden_states,
        max_num_global_attn_indices,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_index_masked,
    ):
        batch_size, seq_len  = hidden_states.shape[:2]

        # prepare global hidden states
        global_attn_hidden_states = hidden_states.new_zeros(batch_size, max_num_global_attn_indices,  self.embed_dim)
        
        global_attn_hidden_states[is_local_index_global_attn_nonzero] = hidden_states[is_index_global_attn_nonzero]

        # global: key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= math.sqrt(self.head_dim)

        # reshape
        global_query_vectors_only_global = (
            global_query_vectors_only_global.contiguous()
            .view(batch_size * self.num_heads, max_num_global_attn_indices,  self.head_dim)
            
        )  # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_key_vectors = (
            global_key_vectors.contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        )  # batch_size * self.num_heads, seq_len, head_dim)
        global_value_vectors = (
            global_value_vectors.contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        )  # batch_size * self.num_heads, seq_len, head_dim)

        # compute attn scores
        global_attn_scores = torch.bmm(global_query_vectors_only_global, global_key_vectors.transpose(1, 2))

        assert list(global_attn_scores.size()) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            seq_len,
        ], (
            "global_attn_scores have the wrong size. Size should be"
            f" {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is"
            f" {global_attn_scores.size()}."
        )

        global_attn_scores = global_attn_scores.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked.squeeze(dim=-1).unsqueeze(dim=1).unsqueeze(dim=1),
            -float("inf"),
        )

        global_attn_scores = global_attn_scores.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)

        # compute global attn probs
        global_attn_probs_float = nn.functional.softmax(
            global_attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        global_attn_probs = nn.functional.dropout(
            global_attn_probs_float.type_as(global_attn_scores), p=self.dropout, training=self.training
        )

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)

        assert list(global_attn_output.size()) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            self.head_dim,
        ], (
            "global_attn_output tensor has the wrong size. Size should be"
            f" {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is"
            f" {global_attn_output.size()}."
        )

        global_attn_probs = global_attn_probs.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_output = global_attn_output.view(
            batch_size, max_num_global_attn_indices, self.num_heads*self.head_dim
        )
        return global_attn_output, global_attn_probs