import torch
import numpy as np
import helper


# load in data
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)
print(text[:100])
print(type(text))

# explore the training data
view_line_range = (0, 10)
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

words = text.split(' ')
print('Roughly the number of total words: {}'.format(len(words)))
print('Roughly the number of unique words: {}'.format(len(set(words))))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))

# create look-up table/dictionary for words
def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_unique = set(text)
    vocab_to_int = {word : i for i, word in enumerate(word_unique)}
    int_to_vocab = {vocab_to_int[word] : word for word in word_unique}
    # return tuple
    return (vocab_to_int, int_to_vocab)

# tokenize punctuation
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    dict_punc = {'.' : '||Period||', ',' : '||Comma||', '"' : '||QuotationMark||',
                 ';' : '||Semicolon||', '!' : '||ExclamationMark||', '?' : '||QuestionMark||',
                 '(' : '||LeftParentheses||', ')' : '||RightParentheses||', '-' : '||Dash||',
                 '\n' : '||Return||'}
        
    return dict_punc

# pre-process training data and save data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

# load the processed data
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


# Build the Neural Network

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')


# batching the data
from torch.utils.data import TensorDataset, DataLoader

def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    n_sequence = len(words) - sequence_length
    
    data = np.zeros((n_sequence, sequence_length))
    for i in range(n_sequence):
        data[i, :] = words[i : i + sequence_length]
        
    target = np.array(words[sequence_length :])
    
    feature_tensor = torch.from_numpy(data).type(torch.LongTensor)
    target_tensor = torch.from_numpy(target).type(torch.LongTensor)
    
    loader = TensorDataset(feature_tensor, target_tensor)
    Data_loader = torch.utils.data.DataLoader(loader, batch_size = batch_size, shuffle = True)
    
    # return a dataloader
    return Data_loader


# test dataloader
test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

data_iter = iter(t_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)

# Create RNN architecture
import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        
        # set class variables
        self.input_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_prop = dropout
        
        # define model layers
        self.embed = nn.Embedding(self.input_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, batch_first = True, dropout = self.dropout_prop)
        self.dropout = nn.Dropout(self.dropout_prop)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        
        
    
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """ 
        out_embed = self.embed(nn_input)
        out_lstm, hidden = self.lstm(out_embed, hidden)
        out_lstm = self.dropout(out_lstm)
        out_lstm = out_lstm.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out_lstm)
        
        out = out.view(nn_input.shape[0], -1, self.output_size)
        # get last batch
        out = out[:, -1]
        

        # return one batch of output word scores and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        
        # initialize hidden state with zero weights, and move to GPU if available
        
        weight = next(self.lstm.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden


# Compute loss 
def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    
    # move data to GPU, if available
    if train_on_gpu:
        inp, target = inp.cuda(), target.cuda()
    
    rnn.zero_grad()
    output, hidden = rnn(inp, hidden)
    loss = criterion(output, target)
    
    loss.backward()
    
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden


# Train and valid model loop
import time as time

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    valid_loss_min = np.Inf
      
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        start = time.time()
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
                 
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
                
            hidden = tuple([each.data for each in hidden])
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)   
            
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []
                print('step {}: {} patches take {}'.format(batch_i, show_every_n_batches, time.time() - start) + 'seconds')
                start = time.time()
                
        rnn.eval() 
        valid_losses = []
        hidden_valid = rnn.init_hidden(batch_size)
        #hidden_valid = tuple([each.data for each in hidden])
        for batch_i, (inputs, target) in enumerate(valid_loader, 1):
            
            n_batches = len(valid_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
                
            hidden_valid = tuple([each.data for each in hidden_valid])
            
            if train_on_gpu:
                inputs, target = inputs.cuda(), target.cuda()
    
            output, hidden_valid = rnn(inputs, hidden_valid)
        
            loss = criterion(output, target)
            
            # record loss
            valid_losses.append(loss.item())

            # printing loss stats
            if batch_i % 50 == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(valid_losses)))
                batch_losses = []
                print('step {}: {} patches take {}'.format(batch_i, show_every_n_batches, time.time() - start) + 'seconds')
                start = time.time()
                
        if np.average(valid_losses) < valid_loss_min:
            valid_loss_min = np.average(valid_losses)
            print('Validation loss decreases, saving the network paramters...')
            torch.save(rnn.state_dict(), 'myTrained_rnn.pt')
        
        rnn.train()
            

    # returns a trained rnn
    return rnn

# hyperparameters
# Sequence Length
sequence_length = 5 # of words in a sequence
# Batch Size
batch_size = 512

# data loader - do not change: No validation
#train_loader = batch_data(int_text, sequence_length, batch_size)

# split data into training data and validation data
val_fra = 0.15
indx = int(np.floor(len(int_text)*(1 - val_fra)))
train_loader = batch_data(int_text[:indx], sequence_length, batch_size)
valid_loader = batch_data(int_text[indx:], sequence_length, batch_size)

# Training parameters
# Number of Epochs
num_epochs = 30
# Learning Rate
learning_rate = 0.001

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int.keys())
# Output size
output_size = vocab_size
# Embedding Dimension
embedding_dim = 300
# Hidden Dimension
hidden_dim = 500
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 200


# Training

# create a model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

if train_on_gpu:
    rnn.cuda()

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')