import torchtext
import torch
import torch.nn as nn
from torchtext.data  import Field, TabularDataset, BucketIterator, Iterator
from torchtext.datasets import Multi30k
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import spacy
!python -m spacy download en
!python -m spacy download de

eng = spacy.load('en_core_web_sm')
ger = spacy.load('de_core_news_sm')

def tokenize_ger(text):
  return [tok.text for tok in ger.tokeniser(text)]

def tokenise_eng(text):
  return [tok.text for tok in eng.tokenize(text)]

german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenise_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, valid_data,test_data = Multi30k.splits(
    exts=(".de",".en"), fields=(german, english),root="data"
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(self,
               embedding_size,
               src_vocab_size,
               trg_vocab_size,
               src_pad_idx,
               num_heads,
               num_encoder_layers,
               num_decoder_layers,
               forward_expansion,
               dropout,
               max_len,
               device,
    ):
        super(Transformer,self).__init__()
        self.src_word_embedding = nn.Embeding(src_vocab_size, embedding_size)
        self.src_position_embeding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embeding = nn.Embedding(trg_vocab_size, embedding,size)
        self.trg_position_embeding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )



        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx


    def make_src_mask(self,src):
        src_mask = src.transpose(0,1) == self.src_pad_idx
        return src_mask
    
    def forward(self,src,trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arrange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device)
        )

        trg_positions = (
            torch.arrange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device)
        )

        embed_src = self.dropout(
            self.src_word_embedding(src) + self.src_position_embedding(src_positions) 
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg = self.transformer.generate_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transgormer(
            embed_src,
            embed_trg,
            src_key_embedding_mask = src_padding_mask,
            tgt_mask = trg_mask,
        )
        out = self.fc_out(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
save_model = True

num_epochs = 5
learning_rate = 3e-4
batch_size = 32

src_vocab_size = 0 # len(german.vocab)
trg_vocab_size = 0 # len(english.vocab)
embeding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
forward_expansion = 2048
dropout = 0.10
max_len = 100
src_pad_idx = 0 #english.vocab.stoi["<pad>"]

writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    datasets= [0,0,0], #(train_data, valid_data, test_data),
    batch_size = batch_size,
    sort_within_batch = True,
    sort_key = lambda x : len(x.src),
    device=device,
)

model = Transformer(
    embeding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("checkpoint.pth.ptar"), model, optimizer)

sentence = "Guten Tag"

for epoch in range(num_epochs):
    print(f"{epoch}/{num_epochs}")

    if save_model:
        checkpoint = {
            state_dict : model.state_dict(),
            optimizer : stat_dict.optimizer(),
        }

        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model,sentence,german,english,device, max_length = 100,
    )

    print(f"translated sentence : {translated_sentence}")
    model.train()

    for batch_idx,batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target[:-1])
        output = output.reshape(-1,output_shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm = 1)

        optimizer.step()

        writer.add_scalar("Training Loss",loss,global_step = step)
        step = step+1
