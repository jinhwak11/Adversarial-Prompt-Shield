import torch
from transformers import DistilBertModel, DistilBertTokenizer
from huggingface_hub import PyTorchModelHubMixin
class DistilBERTClassifier(torch.nn.Module,PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert/distilbert-base-cased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        return output


class APSClassifier():
    def __init__(self,model_path,use_auth_token, device='cuda'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path, use_auth_token = use_auth_token)
        self.cls_model = DistilBERTClassifier.from_pretrained(model_path,use_auth_token=use_auth_token).to(device).eval()
        self.device =device

    def predict(self,dialogue):
        inputs = self.tokenizer.encode_plus(
            dialogue,
            None,
            add_special_tokens=True,
            max_length=self.tokenizer.model_max_length,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        test_data= {
            'ids': torch.tensor(ids, dtype=torch.int),
            'mask': torch.tensor(mask, dtype=torch.int),
            'targets': torch.tensor(4, dtype=torch.int)
        }
        with torch.no_grad():
            ids = test_data['ids'].to(self.device, dtype = torch.long)
            mask = test_data['mask'].to(self.device, dtype = torch.long)
            targets = test_data['targets'].to(self.device, dtype = torch.long)
            outputs = self.cls_model(ids, mask)
            return torch.round(outputs[0])