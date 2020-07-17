import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from configuration_bert import BertConfig
from bert_model_base import BertPreTrainedModel
from activations import gelu, mish, swish

logger = logging.getLogger(__name__)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, "mish": mish}


class BertLayerNorm(nn.Module):
    def __init__(self, n_state, eps=1e-5):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(n_state))
        self.beta = nn.Parameter(torch.zeros(n_state))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class BertEmbedding(nn.Module):
    """Word embeddings, Position embeddings and token_type embeddings"""

    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # input_ids: batch_size x seq_len
        # input_embeds: batch_size x seq_len x embed_size
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        # input_shape: batch_size x seq_len
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze_(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)  # bs(batch_size) x ls(len_seq) x heads x hs(head_size)
        return x.permute(0, 2, 1, 3)  # bs(batch_size) x heads x ls x hs

    def forward(self, hidden_states, attention_mask, head_mask,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False):
        # hidden_states: batch_size x len_seq x num_embed
        mixed_query_layer = self.query(hidden_states)  # batch_size x len_seq x all_head_size
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)  # batch_size x len_seq x all_head_size
            mixed_value_layer = self.value(encoder_hidden_states)  # batch_size x len_seq x all_head_size
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        # bs x heads x ls x ls
        if attention_mask is not None:
            # ? Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        # attention_probs: batch x num_heads x ls x ls

        if head_mask is not None:
            # head_mask: batch x num_heads x ls x ls
            # if None: None
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # bs x heads x ls x hs
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # bs x ls x heads x hs
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # bs x ls x hidden_size
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    # Multi-Head Attention
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask,
                                 output_attentions=output_attentions)
        attention_outputs = self.output(self_outputs[0], hidden_states)
        outputs = (attention_outputs,) + self_outputs[1:]  # (attention probs, ) if exists
        return outputs


class BertIntermediate(nn.Module):
    # Position-wise Feed-Forward
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states  # bs x ls x intermediate_size


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states: bs x ls x intermediate_size
        # input_tensor: bs x ls x hidden_size
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask,
                                                output_attentions=output_attentions)
        # bs x ls x num_embed ==> bs x ls x hidden_states
        # (context_layer, attention_probs)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate = self.intermediate(attention_output)
        layer_output = self.output(intermediate, attention_output)  # bs x ls x hidden_size
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, output_hidden_states=False):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                # gradient_checkpoint: a way to train large model with limited GPU memory
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states,
                                                                  attention_mask, head_mask[i], encoder_hidden_states,
                                                                  encoder_attention_mask, )

            else:
                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, head_mask=head_mask[i],
                                             encoder_hidden_states=encoder_hidden_states,
                                             encoder_attention_mask=encoder_attention_mask,
                                             output_attentions=output_attentions, )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1:],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states (inputs, each_layer_hidden) ), (all attentions)


class BertPooler(nn.Module):
    """Pool the model using the hidden state corresponding to the first token <CLS> of each sentence."""

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]  # bs x ls x hidden_size ==> bs x hidden_size
        pooled_out = self.activation(self.dense(first_token_tensor))
        return pooled_out


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.LayerNorm(self.transform_act_fn(self.dense(hidden_states)))
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.decoder(self.transform(hidden_states))
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertModel(BertPreTrainedModel):
    """This model can behave as an encoder as well as a decoder
        Not complete part: prune heads
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddins(self, value):
        self.embeddings.word_embeddins = value

    def _prune_heads(self):
        # TODO
        pass

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                output_hidden_states=False):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = encoder_hidden_states.shape[:2]
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids,
                                           position_ids=position_ids, inputs_embeds=inputs_embeds)

        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask,
                                       output_attentions=output_attentions, output_hidden_states=output_hidden_states)

        sequence_output = encoder_outputs[0]  # last layer hidden states
        pooled_output = self.pooler(sequence_output)  # bs x hidden_states [take out the first token]

        return (sequence_output, pooled_output,) + encoder_outputs[1:]
        # last-layer hidden state, pooled_output(bs x hidden_states), (all hidden states (inputs, each_layer_hidden)
        # ), (all attentions)


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                labels=None,
                next_sentence_label=None,
                **kwargs):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]
        # prediction_scores: bs x ls x vocabs
        # seq_relationship_score: bs x 2

        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score, next_sentence_label.view(-1))
            total_loss = (masked_lm_loss + next_sentence_loss)
            outputs = (total_loss,) + outputs

        return outputs
        # total_loss, prediction_scores, seq_relationship_score, (all hidden states (inputs, each_layer_hidden)
        # ), (all attentions)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BertLMHeadModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertLMHeadModel, self).__init__(config)
        assert config.is_decoder, "'BertLMHeadModel' needs 'is_decoder=True'."

        self.bert = BertModel(config)  # behave as a decoder
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                labels=None,
                **kwargs):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, )

        # last-layer hidden state, pooled_output(bs x hidden_states), (all hidden states (inputs, each_layer_hidden)
        # ), (all attentions)

        sequence_output = outputs[0]  # bs x ls x hs
        prediction_scores = self.cls(sequence_output)  # bs x ls x vocab_size

        outputs = (prediction_scores,) + outputs[2:]

        # last-layer hidden state, (all hidden states (inputs, each_layer_hidden)), (all attentions)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            prediction_scores = prediction_scores[:, :-1, :].contiguous()  # shift!
            labels = labels[:, 1:].contiguous()  # shift!
            lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (lm_loss,) + outputs
        # lm_loss, prediction_scores, (all hidden states (inputs, each_layer_hidden)
        # ), (all attentions)

        return outputs

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        assert (not config.is_decoder), "'BertForMastedLM' needs config.is_decoder=False"

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                labels=None,
                **kwargs):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, )

        sequence_output = outputs[0]  # bs x ls x hs
        prediction_scores = self.cls(sequence_output)  # bs x ls x vocab_size

        outputs = (prediction_scores,) + outputs[2:]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # no shifts needed
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        input_shape = input_ids.shape
        batch_size = input_shape[0]

        # add a dummy token ?
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # attention_mask: [bs x ls] -> [bs x (ls + 1)]
        dummy_token = torch.full((batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device)

        input_ids = torch.cat([input_ids, dummy_token], dim=-1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    def forward(self, input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                next_sentence_label=None,
                **kwargs):
        """
        example:
        from tokenization_bert import *

        import wget

        url = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-uncased']
        if not os.path.exists(url.split('/')[-1]):
            vocab_file = wget.download(url)  # download
        else:
            vocab_file = url.split('/')[-1]
            bert_tokenizer = BertTokenizer(vocab_file=vocab_file)

        prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        encoding = bert_tokenizer.encode(prompt, next_sentence, return_tesors=True, encode_plus=True)

        from configuration_bert import BertConfig
        import torch
        import torch.utils.checkpoint

        config = BertConfig()
        config.is_decoder = False
        model = BertForNextSentencePrediction(config)
        loss, logits = model(**encoding, next_sentence_label=torch.LongTensor([1]))
        """

        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, )
        # outputs: last-layer hidden state, pooled_output(bs x hidden_states), (all hidden states (inputs,
        # each_layer_hidden) ), (all attentions)

        pooled_output = outputs[1]  # (bs x hidden_states)
        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]

        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            nsp_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (nsp_loss,) + outputs

        return outputs  # (nsp_loss), seq_relationship_score, (all hidden_states), (all attentions)


class BertForSequenceClassification(BertPreTrainedModel):
    """a head for text classification"""

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                attention_mask=None,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                labels=None,
                **kwargs):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, )

        # outputs: last-layer hidden state, pooled_output(bs x hidden_states), (all hidden states (inputs,
        # each_layer_hidden) ), (all attentions)
        pooled_output = outputs[1]
        logits = self.classifier(self.dropout(pooled_output))

        outputs = (logits,) + outputs[2:]

        # logits: bs x num_labels
        if labels is not None:
            if self.num_labels == 1:
                # regression-like, e.g., rating
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))

            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (loss,) + outputs
        return outputs  # (loss, logit, all_hidden_states, all_attentions)


class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=False,
            output_hidden_states=False,
    ):
        # num_choices = input_ids.shape[1] if in
        pass


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        assert config.num_labels, "config.num_labels not exists"
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=False,
            output_hidden_states=False,
    ):
        """
        example:
        from tokenization_bert import *

        import wget

        url = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-uncased']
        if not os.path.exists(url.split('/')[-1]):
            vocab_file = wget.download(url)  # download
        else:
            vocab_file = url.split('/')[-1]
            bert_tokenizer = BertTokenizer(vocab_file=vocab_file)

        text = "Hello, my dog is cute"
        encoding = bert_tokenizer.encode(text, return_tesors=True, encode_plus=True)

        from configuration_bert import BertConfig
        import torch
        import torch.utils.checkpoint

        config = BertConfig()
        config.num_labels = 2
        model = BertForTokenClassification(config)
        labels = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 1]])
        outputs = model(**encoding, labels=labels)
        """
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # bs x ls x num_labels

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if attention_mask is not None:
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                losses_index = attention_mask.view(-1) == 1
                loss = losses[losses_index].sum() / losses_index.sum()
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=False,
            output_hidden_states=False,
    ):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, )
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)  # bs x ls x 2
        start_logits, end_logits = logits.split(1, dim=-1)  # bs x ls x 1
        start_logits = start_logits.squeeze(-1)  # bs x ls
        end_logits = end_logits.squeeze(-1)  # bs x ls

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            pass
        return outputs


if __name__ == '__main__':
    from tokenization_bert import *

    import wget

    url = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-uncased']
    if not os.path.exists(url.split('/')[-1]):
        vocab_file = wget.download(url)  # download
    else:
        vocab_file = url.split('/')[-1]
        bert_tokenizer = BertTokenizer(vocab_file=vocab_file)

    text = "Hello, my dog is cute"
    encoding = bert_tokenizer.encode(text, return_tesors=True, encode_plus=True)

    from configuration_bert import BertConfig
    import torch
    import torch.utils.checkpoint

    config = BertConfig()
    # config.num_labels = 2
    # model = BertForTokenClassification(config)
    # labels = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 1]])
    # outputs = model(**encoding, labels=labels)

    model_from_pretrained = BertForTokenClassification.from_pretrained('bert-base-uncased')

    inputs = bert_tokenizer.encode(text, return_tesors=True, encode_plus=True)

    labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

    outputs = model_from_pretrained(**inputs, labels=labels)
    loss, scores = outputs[:2]


class BertLMCLSHeadModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertLMCLSHeadModel, self).__init__(config)
        assert config.is_decoder, "'BertLMHeadModel' needs 'is_decoder=True'."
        self.num_labels = config.num_labels
        self.bert = BertModel(config)  # behave as a decoder
        self.cls = BertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                labels=None,
                **kwargs):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, )

        # last-layer hidden state, pooled_output(bs x hidden_states), (all hidden states (inputs, each_layer_hidden)
        # ), (all attentions)

        sequence_output = outputs[0]  # bs x ls x hs
        prediction_scores = self.cls(sequence_output)  # bs x ls x vocab_size

        pooled_output = outputs[1]
        logits = self.classifier(self.dropout(pooled_output))

        outputs = (logits, prediction_scores,) + outputs[2:]

        # last-layer hidden state, (all hidden states (inputs, each_layer_hidden)), (all attentions)
        lm_loss_fct = nn.CrossEntropyLoss()
        # shift!
        prediction_scores = prediction_scores[:, :-1, :].contiguous()
        lm_labels = input_ids[:, 1:].contiguous()  # shift!
        lm_loss = lm_loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))

        if labels is not None:
            if self.num_labels == 1:
                # regression-like, e.g., rating
                cls_loss_fct = nn.MSELoss()
                cls_loss = cls_loss_fct(logits.view(-1), labels.view(-1))
            else:
                cls_loss_fct = nn.CrossEntropyLoss()
                cls_loss = cls_loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            loss = 0.5 * lm_loss + cls_loss
            outputs = (loss,) + outputs
            return outputs

        outputs = (lm_loss,) + outputs
        return outputs

