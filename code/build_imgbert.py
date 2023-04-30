import torch
import torch.nn as nn

class ImgBERT(nn.Module):

    def __init__(self, bert, num_labels, image_embeddings_size):
        """
        :param bert: A pre-trained BERT model
        :param num_labels: The number of outputs of the classifier
        :param image_embeddings_size: The size of the image embeddings.
        """
        super(ImgBERT, self).__init__()
        self.bert = bert
        self.num_labels = num_labels

        # Dropout layer
        self.dropout = nn.Dropout(0.2)
        # Reshape image embedding so it can be concatenated with the CLS token
        #self.reshape_emb = nn.Linear(1024, 768)
        self.reshape_emb = nn.Linear(image_embeddings_size, 768)
        # A linear layer as the classifier
        self.classifier = nn.Linear(768*2, num_labels)

    def forward(self, sent_id, mask, img_emb):
        """
        :param sent_id: Ids of input sequence tokens for BERT
        :param mask: Attention mask for BERT
        :param img_emb: Embeddings of the images
        :return: The logits of the classifier and the CLS token from BERT
        """
        outputs = self.bert(sent_id, attention_mask=mask)
        cls = outputs.pooler_output
        x1 = self.dropout(cls)

        x2 = self.reshape_emb(img_emb)
        x = torch.cat((x1, x2), 1)
        y = self.classifier(x)

        return y, cls