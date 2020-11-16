#import trax
#import trax.layers as tl
import json

# Upload word2ind dict
a_file = open("vocab.json", "r")
word2ind = a_file.read()


def classifier(vocab_size=len(word2ind), embedding_dim=2048, output_dim=2, mode='train'):
        
### START CODE HERE (Replace instances of 'None' with your code) ###
    # create embedding layer
    embed_layer = tl.Embedding(
        vocab_size=vocab_size, # Size of the vocabulary
        d_feature=embedding_dim)  # Embedding dimension
    
    # Create a mean layer, to create an "average" word embedding
    mean_layer = tl.Mean(axis = 1, keepdims= False)

    lstm = tl.LSTM(embedding_dim)
    
    # Create a dense layer, one unit for each output
    dense_output_layer = tl.Dense(n_units = output_dim)

    
    # Create the log softmax layer (no parameters needed)
    log_softmax_layer = tl.LogSoftmax()
    
    # Use tl.Serial to combine all layers
    # and create the classifier
    # of type trax.layers.combinators.Serial
    model = tl.Serial(
      embed_layer, # embedding layer
      mean_layer, # mean layer
      dense_output_layer, # dense output layer 
      log_softmax_layer # log softmax layer
    )
### END CODE HERE ###     
    
    # return the model of type
    return model

def tweet_to_tensor(tweet, unknown , vocab_dict) : 
  """
   This function trasnform a tweet to a tensor
   Input : 
   tweet  : String
   unknown : symbol of unknown word
   vocab_dict :  a dictionary which map word to indice
   output : 
   tensor  : list/array
   """
  tokenized = process_tweet(tweet)

  unk_id = vocab_dict[unknown]

  tensor = []
  for token in tokenized : 
    if token in vocab_dict.keys() : 
      tensor.append(vocab_dict[token])
    else : 
      tensor.append(unk_id)
  
  return tensor
