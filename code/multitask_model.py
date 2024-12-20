import torch
from torch import nn
from transformers import BertModel

class MultiTaskModel(nn.Module):
    """Multi-task learning model for PTDC and LCDC."""
    def __init__(self, hidden_size=768, num_labels_confusion=3, num_labels_post_type=6):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Classification heads for LCDC and PTDC
        self.confusion_head = nn.Linear(hidden_size, num_labels_confusion)
        self.post_type_head = nn.Linear(hidden_size, num_labels_post_type)

        # Shared Attention Layer (SAL)
        self.shared_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)

    def forward(self, input_ids, attention_mask, task='confusion'):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = bert_outputs.last_hidden_state

        # Shared attention for multitask learning
        attn_output, _ = self.shared_attention(last_hidden_state, last_hidden_state, last_hidden_state)
        pooled_output = attn_output.mean(dim=1)

        if task == 'confusion':
            logits = self.confusion_head(pooled_output)
        elif task == 'post_type':
            logits = self.post_type_head(pooled_output)
        else:
            raise ValueError("Invalid task type. Choose from 'confusion' or 'post_type'.")

        return logits
