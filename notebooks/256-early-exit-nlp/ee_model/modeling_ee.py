import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_ee import EarlyExitConfig


def entropy(x):
    softmax_probs = nn.functional.softmax(x, dim=1)
    ee_entropy = -torch.sum(softmax_probs * torch.log(softmax_probs), dim=1)
    return ee_entropy


def activate_ee(logits, threshold):
    return torch.all(entropy(logits) < threshold)


# Taken from HuggingFace transformers
def create_extended_attention_mask(attention_mask, dtype):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    else:
        extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def compute_qa_loss(logits, start_positions, end_positions):
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()
    qa_loss = 0

    if start_positions is not None and end_positions is not None:
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_function = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_function(start_logits, start_positions)
        end_loss = loss_function(end_logits, end_positions)
        qa_loss = (start_loss + end_loss) / 2
    return qa_loss


class RampClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, hidden_state):
        logits = self.classifier(hidden_state)
        return logits


# Trainable model wrapper
class EarlyExitModelForQuestionAnswering(PreTrainedModel):
    config_class = EarlyExitConfig

    def __init__(self, config):
        super().__init__(config)
        orig_model = AutoModel.from_pretrained(config.base_model)

        self.embeddings = orig_model.embeddings
        self.transformer_layers = orig_model.encoder.layer
        self.ramp_classifiers = nn.ModuleList([RampClassifier(config) for _ in range(config.num_hidden_layers)])

        self.ee_entropy = 0.0

    def get_entropy_threshold(self):
        return self.ee_entropy

    def set_entropy_threshold(self, ee_entropy):
        self.ee_entropy = ee_entropy

    def forward(self, input_ids=None, attention_mask=None, start_positions=None, end_positions=None, training=False, training_phase=1, **kwargs):
        embeddings_output = self.embeddings(input_ids=input_ids)

        # required for back-propagation
        attention_mask = create_extended_attention_mask(attention_mask, torch.float32)

        layer_input = embeddings_output
        total_loss = 0.0

        for i, transformer_layer in enumerate(self.transformer_layers):
            layer_output = transformer_layer(hidden_states=layer_input, attention_mask=attention_mask)
            layer_logits = self.ramp_classifiers[i](layer_output[0])

            if training:
                if training_phase == 2 and i < len(self.transformer_layers)-1:
                    total_loss += compute_qa_loss(layer_logits, start_positions, end_positions)
                elif training_phase == 1 and i == len(self.transformer_layers)-1:
                    total_loss = compute_qa_loss(layer_logits, start_positions, end_positions)
            else:
                if activate_ee(layer_logits, self.ee_entropy):
                    print(f'Exit layer: {i+1}')
                    break

            layer_input = layer_output[0]

        start_logits, end_logits = layer_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return QuestionAnsweringModelOutput(loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=None,
            attentions=None,
        )
