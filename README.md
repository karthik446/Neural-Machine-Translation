# NMT Assignment
Natural Machine Translation

# model_embeddings.py
Model Embedding takes in a Embedding size and complete vocabulary. Uses Pytorch Embedding to create source and target embeddings 

# nmt_model.py
There is an encoder and a decoder
Encoder is Bidirectional LSTM Layer with bias
Decoder is a LSTM Cell
Encode method in the model takes in padded source and source lengths which has list of integers with maximum lengths and returns encoder hidden states and decoder initial states.

Decode Method takes in encoder hidden outputs, encoder masks , initial state of the decoder from encode method and target padded tensor. It creates target embeddings and splits on time and loops through it to generate e_t, decoder state, combined output from the step ( attention )

Step method calculates the attention scores, attention probability distribution from the encoder hidden state and decoder hidden state.

decoder hidden state and attention output is concatenated and dropout(tanh())
is applied to the concatenanted vector.

Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository
