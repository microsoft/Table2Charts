from typing import Dict, Tuple

import torch
import torch.nn as nn
from allennlp.modules.attention import LinearAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn import util
from allennlp.nn.activations import Activation
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import GRUCell

from data import TokenType
from .config import CopyNetConfig
from .embedding import InputEmbedding


def get_field_action_mask(actions: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Mark field action tokens from the action space.
    Note that for an action space sequence, field tokens are always the first ones
    (following by command tokens).
    :param actions: Action Space [str -> (batch_size, max_action_space_length, ?)]
    :return: Mask for the action space where only field action tokens are marked as 1.
    """
    # shape: batch_size * max_action_space_length
    field_action_mask = (actions["token_types"] == TokenType.FIELD.value)
    return field_action_mask


def get_target_to_source(state: Dict[str, torch.Tensor], actions: Dict[str, torch.Tensor]):
    """
    Target (state) to source (action space) probability dist.
    In our case each dist has only one non-zero element.
    In other words, it's a one-hot vector.
    :param state: Should have key "field_indices". See to_dict() method of Sequence class.
    :param actions: Action space
    :return: One-hot vectors indicating each timestep of a state is copied from which source field.
    """
    # shape: batch_size * state_len
    index = state["field_indices"]
    batch_size, state_len = index.size()
    src_len = actions["mask"].size(1)
    # shape: state_len * batch_size
    index.transpose_(0, 1)

    # TODO: change to torch.nn.functional.one_hot() after upgrade to torch >= 1.1
    # shape: (src_len + 1) * src_len, the last row is all zero for index = -1
    id_matrix = torch.eye(src_len + 1, src_len, dtype=torch.uint8, device=index.device)
    # shape: state_len * batch_size * src_len
    one_hots = id_matrix[index, :]
    return one_hots


def get_final_step_log_probs(step_log_probs: torch.Tensor, dqn_state_mask: torch.Tensor) -> torch.Tensor:
    """
    :param step_log_probs: batch * dqn_state_length * (command_token_size + source_length) * 2
    :param dqn_state_mask: batch * dqn_state_length
    :return: The output log probabilities after the final step of each action sequence
    """
    # batch * 1 * 1
    prob_index = torch.sum(dqn_state_mask, 1, keepdim=True, dtype=torch.long).unsqueeze(-1) - 1
    # batch * (command_token_size + source_length) * 2
    prob_index = prob_index.expand(-1, step_log_probs.size(-2), step_log_probs.size(-1))
    # batch * 1 * (command_token_size + source_length) * 2
    prob_index = prob_index.unsqueeze(1)
    # batch * 1 * (command_token_size + source_length) * 2
    final_step_log_probs = step_log_probs.gather(1, prob_index)
    # batch * (command_token_size + source_length) * 2
    final_step_log_probs = final_step_log_probs.squeeze(1)
    return final_step_log_probs


def swap_field_cmd_probs(input_probs, num_cmd_tokens, source_mask):
    """
    :param input_probs: batch * (command_token_size + source_length) * 2
    :param num_cmd_tokens: command_token_size
    :param source_mask: batch * source_length
    :return: The same log probabilities but with field tokens first, followed by command tokens.
    """
    batch_size, source_length = source_mask.size()

    # batch * 1
    num_fields = torch.sum(source_mask, 1, keepdim=True, dtype=torch.long)
    # batch * num_cmd_tokens
    cmd_replace_index = num_fields.expand(-1, num_cmd_tokens) +\
                        torch.arange(num_cmd_tokens, device=num_fields.device, dtype=torch.long).repeat(batch_size, 1)

    # batch * source_length
    not_moved_part = torch.zeros(batch_size, source_length, device=source_mask.device, dtype=torch.long)
    not_moved_part.masked_fill_(source_mask.bool().logical_not(), num_cmd_tokens)
    src_replace_index = torch.arange(source_length, device=source_mask.device, dtype=torch.long).repeat(batch_size, 1)
    src_replace_index = src_replace_index + not_moved_part

    # batch * (command_token_size + source_length)
    replace_index = torch.cat((cmd_replace_index, src_replace_index), dim=1)
    # batch * (command_token_size + source_length) * 2
    replace_index = replace_index.unsqueeze(-1).expand(-1, -1, 2)

    # batch * (command_token_size + source_length) * 2
    return_probs = input_probs.new_zeros(input_probs.size())
    return_probs.scatter_(1, replace_index, input_probs)
    return return_probs


class CopyNetSeq2Seq(nn.Module):

    def __init__(self,
                 input_embedding: InputEmbedding,
                 config: CopyNetConfig) -> None:
        super().__init__()

        self.data_len = config.data_len
        # Encoding modules.
        self._encoder = PytorchSeq2SeqWrapper(
            torch.nn.GRU(input_size=config.hidden, hidden_size=config.encoder_GRU_hidden, num_layers=config.encoder_layers,
                         bidirectional=True, batch_first=True))
        # Embedding modules.
        self.input_embed = input_embedding
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        # self.encoder_output_dim = self._encoder.get_output_dim()  # = config.encoder_GRU_hidden * 2
        self.encoder_output_dim = config.encoder_GRU_hidden * 2
        self.decoder_input_dim = config.decoder_hidden_size
        self.decoder_output_dim = config.decoder_GRU_hidden  # = config.decoder_GRU_hidden * 2

        # Reduce dimensionality of encoder output to reduce the number of decoder parameters.
        self.encoder_output_projection = Linear(self.encoder_output_dim, self.decoder_output_dim)

        # The decoder input will be a function of the embedding of the previous predicted token,
        # an attended encoder hidden state called the "attentive read", and another
        # weighted sum of the encoder hidden state called the "selective read".
        # While the weights for the attentive read are calculated by an `Attention` module,
        # the weights for the selective read are simply the predicted probabilities
        # corresponding to each token in the source sentence that matches the target
        # token from the previous timestep.
        self._attention = LinearAttention(self.decoder_output_dim, self.decoder_output_dim,
                                          activation=Activation.by_name('tanh')())
        # config.hidden * 2: bidirectional
        self._input_projection_layer = Linear(
            config.feature_dim + self.decoder_output_dim * 2,
            self.decoder_input_dim)

        # We then run the projected decoder input through an LSTM cell to produce
        # the next hidden state.
        # self._decoder_cell = LSTM(self.decoder_input_dim, self.decoder_output_dim,
        # num_layers=self.num_layers, batch_first=True)
        self._decoder_cell = GRUCell(self.decoder_input_dim, self.decoder_output_dim)
        self._command_token_size = config.num_cmd_tokens

        # We create a "generation" score for each token in the target vocab
        # with a linear projection of the decoder hidden state.
        self._output_generation_layer_1 = Linear(self.decoder_output_dim, self._command_token_size)
        self._output_generation_layer_2 = Linear(self.decoder_output_dim, self._command_token_size)

        # We create a "copying" score for each source token by applying a non-linearity
        # (tanh) to a linear projection of the encoded hidden state for that token,
        # and then taking the dot product of the result with the decoder hidden state.
        self._output_copying_layer_1 = Linear(self.decoder_output_dim, self.decoder_output_dim)
        self._output_copying_layer_2 = Linear(self.decoder_output_dim, self.decoder_output_dim)

        self._softmax = nn.LogSoftmax(dim=-1)

    def forward(self, dqn_state, dqn_actions) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        State and the action space

        Returns
        -------
        Action values
        """
        # Trim tensors in dqn_actions by deleting the last num_cmd_tokens columns.
        max_num_fields = dqn_actions["mask"].size()[1] - self._command_token_size
        # There will be faked inputs for empty sample batch from student._feed_batch_nn_(),
        # which ignores the output of DQN model.
        if max_num_fields < 0:
            return dqn_actions["mask"].new_zeros((1, 2))

        # When ablation, dqn_state["semantic_embeds"] size will be like [batch], and useless in input_embed,
        # so it will be "None" in dqn_actions
        dqn_actions = {key: None if len(dqn_actions[key].size()) == 1 else torch.narrow(dqn_actions[key], 1, 0, max_num_fields) for key in dqn_actions.keys()}

        self._encoder._module.flatten_parameters()
        state = self._encode(dqn_actions)

        # B * len_x * embed_hidden
        x = self.input_embed(dqn_state["token_types"], dqn_state["segments"],
                             dqn_state["semantic_embeds"], dqn_state["categories"])
        # B * len_x * hidden
        if self.data_len > 0:
            x = torch.cat((x, dqn_state["data_characters"]), -1)

        state = self._init_decoder_state(state)
        target_to_source_total = get_target_to_source(dqn_state, dqn_actions).float()
        return self._forward_estimation(x, dqn_state["mask"], target_to_source_total, state)

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        batch_size, _ = state["source_mask"].size()

        # Initialize the decoder hidden state with the final output of the encoder,
        # and the decoder context with zeros.
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            state["encoder_outputs"],
            state["source_mask"],
            self._encoder.is_bidirectional())
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        # state["decoder_context"] = state["encoder_outputs"].new_zeros(batch_size, self.decoder_output_dim)

        return state

    def _encode(self, dqn_actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode source input sentences.
        """
        # shape: (batch_size, max_num_fields)
        source_mask = get_field_action_mask(dqn_actions)

        y = self.input_embed(dqn_actions["token_types"], dqn_actions["segments"],
                             dqn_actions["semantic_embeds"], dqn_actions["categories"])
        # shape: B * max_num_fields * hidden
        if self.data_len > 0:
            y = torch.cat((y, dqn_actions["data_characters"]), -1)

        # shape: batch_size * max_num_fields * config.encoder_GRU_hidden * 2
        encoder_outputs = self._encoder(y, source_mask)
        # shape: batch_size * max_num_fields * decoder_output_dim
        encoder_outputs = self.encoder_output_projection(encoder_outputs)

        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _decoder_step(self,
                      last_embedded: torch.Tensor,
                      selective_weights: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (group_size, max_input_sequence_length)
        encoder_outputs_mask = state["source_mask"].float()
        # shape: (group_size, target_embedding_dim)
        embedded_input = last_embedded
        # shape: (group_size, max_input_sequence_length)
        attentive_weights = self._attention(
            state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask)
        # shape: (group_size, encoder_output_dim)
        attentive_read = util.weighted_sum(state["encoder_outputs"], attentive_weights)
        # shape: (group_size, encoder_output_dim)
        selective_read = util.weighted_sum(state["encoder_outputs"], selective_weights)
        # shape: (group_size, target_embedding_dim + encoder_output_dim * 2)
        decoder_input = torch.cat((embedded_input, attentive_read, selective_read), -1)
        # shape: (group_size, decoder_input_dim)
        projected_decoder_input = self._input_projection_layer(decoder_input)

        # GRU
        state["decoder_hidden"] = self._decoder_cell(projected_decoder_input, state["decoder_hidden"])

        # LSTM
        # state["decoder_hidden"], state["decoder_context"] = self._decoder_cell(
        #         projected_decoder_input,
        #         (state["decoder_hidden"], state["decoder_context"]))
        # state["decoder_hidden"] = state["decoder_hidden"][self.decoder_layers-1].squeeze(0)
        # state["decoder_context"] = state["decoder_context"][self.decoder_layers-1].squeeze(0)
        return state

    def _get_generation_scores(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._output_generation_layer_1(state["decoder_hidden"]),\
               self._output_generation_layer_2(state["decoder_hidden"])

    def _get_copy_scores(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, decoder_output_dim)
        copy_projection_1 = self._output_copying_layer_1(state["encoder_outputs"])
        # shape: (batch_size, max_input_sequence_length, decoder_output_dim)
        copy_projection_1 = torch.tanh(copy_projection_1)
        # shape: (batch_size, max_input_sequence_length)
        copy_scores_1 = copy_projection_1.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1)
        # shape: (batch_size, max_input_sequence_length, decoder_output_dim)
        copy_projection_2 = self._output_copying_layer_2(state["encoder_outputs"])
        # shape: (batch_size, max_input_sequence_length, decoder_output_dim)
        copy_projection_2 = torch.tanh(copy_projection_2)
        # shape: (batch_size, max_input_sequence_length)
        copy_scores_2 = copy_projection_2.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1)
        return copy_scores_1, copy_scores_2

    def _forward_estimation(self,
                            dqn_state_embeded: torch.Tensor,
                            dqn_state_mask: torch.Tensor,
                            target_to_source_total: torch.Tensor,
                            state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the loss against gold targets.
        dqn_state_embeded: batch * len_x * hidden
        dqn_state_mask: batch * len_x
        target_to_source_total: len_x * batch * len_y
        """
        dqn_state_length = dqn_state_embeded.size(1)
        num_decoding_steps = dqn_state_length

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        step_log_probs_lists = []
        for timestep in range(num_decoding_steps):
            # shape: (batch_size, hidden)
            input_embedded = dqn_state_embeded[:, timestep]

            # We need to keep track of the probabilities assigned to tokens in the source
            # sentence that were copied during the previous timestep, since we use
            # those probabilities as weights when calculating the "selective read".
            # shape: (batch_size, source_length)
            selective_weights = target_to_source_total[timestep]

            state = self._decoder_step(input_embedded, selective_weights, state)

            # Get generation scores for each token in the command_token.
            # shape: (batch_size, command_token_size)
            generation_scores_1, generation_scores_2 = self._get_generation_scores(state)

            # Get copy scores for each token in the source sentence, excluding the start
            # and end tokens.
            # shape: (batch_size, source_length)
            copy_scores_1, copy_scores_2 = self._get_copy_scores(state)

            # Concat un-normalized generation and copy scores.
            # shape: (batch_size, command_token_size + source_length)
            all_scores_1 = torch.cat((generation_scores_1, copy_scores_1), dim=-1)
            all_scores_2 = torch.cat((generation_scores_2, copy_scores_2), dim=-1)
            # shape: (batch_size, command_token_size + source_length, 2)
            all_scores = torch.stack((all_scores_1, all_scores_2), dim=-1)
            log_probs = self._softmax(all_scores)

            # shape: (batch_size, 1, command_token_size + source_length, 2)
            step_log_probs_lists.append(log_probs.unsqueeze(1))

        # shape: (batch_size, dqn_state_length, command_token_size + source_length, 2)
        step_log_probs = torch.cat(step_log_probs_lists, dim=1)

        # (batch_size, command_token_size + source_length, 2)
        final_step_log_probs = get_final_step_log_probs(step_log_probs, dqn_state_mask)
        # (batch_size, source_length + command_token_size, 2)
        final_step_log_probs = swap_field_cmd_probs(final_step_log_probs, self._command_token_size, source_mask)

        return final_step_log_probs

    def get_embed_parameters(self):
        return self.input_embed.parameters()

    def get_encoder_parameters(self):
        return self._encoder.parameters()

    def get_decoder_modules(self):
        decoder_modules = [
            self._attention, self._input_projection_layer, self._decoder_cell,
            self._output_generation_layer_1, self._output_generation_layer_2,
            self._output_copying_layer_1, self._output_copying_layer_2
        ]
        return decoder_modules

    def get_embed_modules(self):
        return [self.input_embed]

    def get_encoder_modules(self):
        return [self._encoder, self.encoder_output_projection]


class CopyNet(CopyNetSeq2Seq):
    def __init__(self, config: CopyNetConfig):
        input_embed = InputEmbedding(config)
        super().__init__(input_embed, config)


# if __name__ == "__main__":
#     step_log_probs = torch.arange(0.0, 168.0, 1.0).view(2, 6, 7, 2)
#     mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
#     final_step_log_probs = get_final_step_log_probs(step_log_probs, mask)
#     assert final_step_log_probs.size() == (2, 7, 2)
#
#     num_cmd_tokens = 3
#     source_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
#     final_step_log_probs = swap_field_cmd_probs(final_step_log_probs, num_cmd_tokens, source_mask)
#     assert final_step_log_probs.size() == (2, 7, 2)
