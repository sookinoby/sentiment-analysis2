# Text classifcation for sentiment analysis using Mxnet

In this notebook, we are going to classify sentiment by building a neural network using mxnet.  The neural network will take a movie review as input and tries to identify the if the movie express a positive or negative opinion about the movie. We will start with simple dense model and then build model similar to (Yoon Kim's)[https://arxiv.org/abs/1408.5882] paper We will also visualize the output using tnse(visualization techniques for high dimensional data). Finally, we will use transfer learning to use pre-built embedding(glove)[https://nlp.stanford.edu/projects/glove/] in our neural network to classify sentences.

Although there are many deep learning frameworks (TensorFlow, Keras, Torch, Caffee), mxnet is gaining popularity due to its scalability across multiple GPU.

This notebook expects you to have a basic understanding of convolution operation, neural network, activation units, gradient decent, numpy. 

By the end of the notebook, you will be able to 

1. Understand the complexity of sentiment analysis.
2. Understand embedding and their use.
3. Prepare dataset for training neural network.
4. Implement custom neural network architecture for classifying sentiment using various different models.
5. Visualize the result and understand our model using (t-sne)[https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding]. 
6. Use prebuilt embedding like glove to train on data with constraints(small dataset or small sentences). 

## Sentiment Analysis
Sentiment analysis is a complex task. Understanding if a sentence expresses a positive or negative opinion is very difficult to predict. Let us a take sentence

like this "The movie was unintelligent, gross and vulguar, but I loved it". Although, the sentence contains lot of negative words (unintelligent, gross, vulgar) , the sentiment expressed is positive (but, I loved it). Some sentences can be like "The movie doesnt care about cleverness, wit or any other kind of intelligence."

There can be some sentences which can be sarcastic ("Taken3 makes Taken2 a master piece."). We can understand this sentence expresses a negative review because we understand the context(taken2 was bad). 

Another example would be "an infection is destroying white blood cells" and "white blood cell is destroying an infection". Although, they contain the same words, they convey different sentiments.

Sentiment analysis is a very difficult task due to the context and its accuracy mostly depends on the dataset thats processed. A sentiment analyzer can perform very good on one dataset and poorly on another dataset. The machine learning developer should be aware and check their model to capture the variation in the data. 

Let us consider another sentences -     `

*side note-  Taken3,taken2, taken are english movies. Taken was a enjoyable movie.


## Encoding the dataset
Encoding a image into numbers is fairly straight forward. Each pixel can only take up values from 0-255 (RGB) color space. Resizing of image doesnt affect the content of image. But encoding natural language(words) into numbers is not straight forward. A language can have huge vocabulary and sentences formed by those words can be of varying length. Resizing a sentences can change the meaning of the sentences entriely. Lets consider a toy example which helps us to understand the 
Lets consider a set of words {I, love, cake, hate, pizza } which forms out entire vocabulary.

Lets us consider two sentences formed out of these vocabulary.
"I love cake"  
"I hate pizza"

How can we encode theses sentences in a numbers? One way is to represent the vocabulary like one hot encoded matrix as show below.

In this representation, if there are N words in vocabulary, we need NXN matrix and this matrix can be called as vocabulary matrix. Vocabulary matrix forms the look up table.

Now, lets us try to encode the sentence.
The Setences "I hate pizza" will become a matrix shown below.

 If the word is present in the sentence, then corresponding row of the  vocabulary matrix is copied. This is the same operation that is performed by the embedding layer of the mxnet(or any other deep learning framework).
 (Embedding layer)[http://mxnet.io/api/python/symbol.html#mxnet.symbol.Embedding] just peforms the look up operations and should not be confused with (word-embedding)[https://en.wikipedia.org/wiki/Word_embedding].

Can we do better than one-code encoding the vocabulary?. Is there a better way to represent the words?. Word embedding solves this problem. Instead of discretizing the words, it provides a continous represantation of words. A word embedding matrix can look like this.

Instead of the representing the word as NXN matrix, we represent the words with N*3 matrix, where there 3 is embedding size. So each word can be represented as a 3 dimensional vector, instead of N dimensional vector. Words embedding not only reduces the size of representation of vocabulary matrix, but tries to bring bring semantic relationship between words. For example,
"pizza" and "cake" have nearly similar word embedding, since both refers to type of food. "Love" and "Hate" have same magnitude in 2nd dimension since they convey feelings  but entriely different magnitude in 1st dimension (0.9 and 0.1 respectively) since they convey opposite sentiments. Embedding words into smaller dimensions can help us in several ways . These embedding can be learnt by deep neural network automatically during sentiment classification. The vector of particular words can be treated as weights that needs to be learnt by deep neural network. The embedding technqiues can be used on images and other data and commonly popularised as (autoencoder)[https://en.wikipedia.org/wiki/Autoencoder] networks.


## Convolutional  on sentences.

Once the sentence has been encoded into matrix using embedding layer, we can perform the using a convulution of size filter_size * embedding. as shown below. This will help to learn n-grams from the setences.


Next we will look into how sentiment classification is performed using MXNet.


## Preparing your environment
If you're working in the AWS Cloud, you can save yourself the installation management by using an [Amazon Machine Image (AMI)](https://aws.amazon.com/marketplace/pp/B01M0AXXQB#support) preconfigured for deep learning. This will enable you to skip steps 1-5 below.  

Note that if you are using a conda environment, remember to install pip inside conda, by typing 'conda install pip' after you activate an environment. This step will save you a lot of problems down the road.

Here's how to get set up: 

1. First, get [Anaconda](https://www.continuum.io/downloads), a package manager. It will help you to install dependent Python libraries with ease.
2. Next, install [scikit learn](http://scikit-learn.org/stable/install.html), a general-purpose scientific computing library. We'll use this preprocess our data. You can install it with 'conda install scikit-learn'.
3. Then grab the Jupyter Notebook, with 'conda install jupyter notebook'.
4. And then, get [MXNet](http://mxnet.io/get_started/install.html), an open source deep learning library.

The next 3 helps in visualizing the word-embedding and is not manadatory. I highly recommend visualizing the results.

5. Next, we need [cython](https://anaconda.org/anaconda/cython) required for bhtnse
6. Next, we need[bhtsne](https://github.com/dominiek/python-bhtsne), A c++, python implementation of
 tnse. Dont use scikit-learn tnse implementation as it crashes the python kernel and reported [here](https://github.com/scikit-learn/scikit-learn/issues/4619)
7. Finally, we need [matplotlib](https://anaconda.org/anaconda/matplotlib) for plots and visualization.  

Here are the commands you need to type inside the anaconda environment (after activation of the environment):
1. conda install pip 
2. pip install opencv-python
3. conda install scikit-learn
4. conda install jupyter notebook
5. pip install mxnet
6. conda install -c anaconda cython
7. pip install bhtsne 
8. conda install -c anaconda matplotlib

## The data set
In order to learn about any deep neural network, we need data. For this notebook, we use a movie review dataset from stanford. Here is the [link](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

The data set consists of 25,000 samples in training set and 25000 samples in test set, with 12500 positive and 12500 negative sentiment in each set. We will only be using the training set that will be randomly split into train,validation and test set. As discussed earlier, sentiment analysis is a complex task and its accuracy depends the dataset. A model should be tested on variety of data before deploying to production.

Here's the code for loading the data. We assign a lable of 1 for positive sentiment and 0 for negative sentiment. We wont be one-hot encoding the labels since will use [softmaxoutput](http://mxnet.io/api/python/symbol.html#mxnet.symbol.SoftmaxOutput) layer of MXnet to perform classification.

In other frame-work, one-hot encoding the labels might be necessary.

```python
import os
def read_files(foldername):
    import os
    sentiments = []
    filenames = os.listdir(os.curdir+ "/"+foldername)
    for file in filenames:
        with open(foldername+"/"+file,"r", encoding="utf8") as pos_file:
            data=pos_file.read().replace('\n', '')
            sentiments.append(data)
    return sentiments
    
#contains positive movie review    
foldername = "easy/pos"
postive_sentiment = read_files(foldername)

#contains negative movie review
foldername = "easy/neg"
negative_sentiment = read_files(foldername)

positive_labels = [1 for _ in postive_sentiment]
negative_labels = [0 for _ in negative_sentiment]
```


## preparing the dataset and encoding
The reviews are cleaned to remove urls, special character and etc. You can also use nltk library to preprocess the text. We use a custom function to clean the data. Once the data is cleaned, we need to form the vocabulary (all the unique words available in dataset). Next, we need to identity the most common words in the review.  This prevents rare words like 'director name,actor name' to influence the outcome of classifier. We also map each words(sorted by descending order based on the fequency of occurence) to unique number and is stored in dictionary called word_dict. We also perform the inverse mapping from the idx to the word. We use a vocabulary of 5000 words. The vocabulary size, sentences length, and embedding dimensions are also parameters and can be experimented while building neural network.

```python
#some string preprocessing
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"'", " ", string)
    string = re.sub(r"\[", " ", string)
    string = re.sub(r"\]", " ", string)
    return string.strip().lower()

#create a dict of word and their count in entrie dataset{word:count}
word_counter = Counter()
def create_count(sentiments):
    idx = 0
    for line in sentiments:
        for word in (clean_str(line)).split():
            if word not in word_counter.keys():               
                word_counter[word] = 1
            else:
                word_counter[word] += 1

#Assigns a unique a number for each word (sorted by descending order based on the frequency of occurrence)
# and returns a word_dict
def create_word_index():
    idx = 0
    word_dict = {}
    for word in word_counter.most_common():
        word_dict[word[0]] = idx
        idx+=1
    return word_dict

#inverse mapping 
idx2word = {v: k for k, v in word_dict.items()}
    
```

Next we actually encode the sentences into numbers using the word_dict. The following code performs the operations.

```python
#Creates a encoded sentences. 
#Assigns the unique id from wordict to the words in the sentences
def encoded_sentences(input_file,word_dict):
    output_string = []
    for line in input_file:
        output_line = []
        for word in (clean_str(line)).split():
            if word in word_dict:
                output_line.append(word_dict[word])
        output_string.append(output_line)
    return output_string

def decode_sentences(input_file,word_dict):
    output_string = []
    for line in input_file:
        output_line = ''
        for idx in line:
            output_line += idx2word[idx] + ' '
        output_string.append(output_line)
    return output_string
```

Next we pad or turncate the sentences to a length of 500 words. We can choose different length based on the sentence statistics.


```python
def pad_sequences(sentences,maxlen=500,value=0):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    for sen in sentences:
        new_sentence = []
        if(len(sen) > maxlen):
            new_sentence = sen[:maxlen]
            padded_sentences.append(new_sentence)
        else:
            num_padding = maxlen - len(sen)
            new_sentence = np.append(sen,[value] * num_padding)
            padded_sentences.append(new_sentence)
    return padded_sentences
```

Next we still the data into train,validation and test set as follows


```python
#train+validation, test split
X_train_val, X_test, y_train_val, y_test_set = train_test_split(t_data, all_labels, test_size=0.3, random_state=42)

#train, validation split of data
X_train, X_val, y_train, y_validation = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)
```

## Building the deepnet

Now, enough of preparing our data set. Let's actually code up the neural network. Building neural networks is something of a black art at this point in history; you never know which can suite your needs. We will start out with a simple dense network and then move on to complex neural network

The neural network code is concise and simple, thanks to MXNet's symbolic API:

```python
#A simple dense model
input_x_1 = mx.sym.Variable('data')
embed_layer_1 = mx.sym.Embedding(data=input_x_1, input_dim=vocab_size, output_dim=embedding_dim, name='vocab_embed')
flatten_1 = mx.sym.Flatten(data=embed_layer_1)
fc1_1 = mx.sym.FullyConnected(data=flatten_1, num_hidden=500,name="fc1")
relu3_1 = mx.sym.Activation(data=fc1_1, act_type="relu" , name="relu3")
fc2_1 = mx.sym.FullyConnected(data=relu3_1, num_hidden=2,name="final_fc")
dense_1 = mx.sym.SoftmaxOutput(data=fc2_1, name='softmax')
```

Let's break down the code a bit. First, it creates a data layer(input layer) that actually holds the dataset while training:

```python 
data = mx.symbol.Variable('data')
```

The vocab_embed layer performs the look up into the embedding matrix (which will be learnt in the process):

```python
embed_layer_1 = mx.sym.Embedding(data=input_x_1, input_dim=vocab_size, output_dim=embedding_dim, name='vocab_embed')
```
The flatten layer flattens the embedding layer weights with dimesion of seq_lenXembedding dimension into a column vector of 1X(seq_len*embedding) which serves the input for next dense layer.
```python
flatten_1 = mx.sym.Flatten(data=embed_layer_1)
```

The fully connected layers connects every neuron(output) from flatten layer(previous layer) to current layer.
```python
mx.sym.FullyConnected(data=flatten_1, num_hidden=500,name="fc1")
```
The relu3_1 layer performs non-linear activation on the input for learning complex functions.
```python
relu3_1 = mx.sym.Activation(data=fc1_1, act_type="relu" , name="relu3")
```

The final dense layers (softmax) performs the classification. The [SoftmaxOutput](http://mxnet.io/api/python/symbol.html#mxnet.symbol.SoftmaxOutput) layer in MXNet performs the one hot encoding of output then applies the softmax function. Please see the example in the documentation.

## Training the network
We are training the network using GPUs, since it's faster. A single pass-through of the training set is referred to as one "epoch," and we are training the network for 3 epochs "num_epoch = 3". We also periodically store the trained model in a JSON file, and measure the train and validation accuracy to see our neural network 'learn.' 

Here is the code: 
```python
#Create Adam optimiser
adam = mx.optimizer.create('adam')

#Checkpointing (saving the model). Make sure there is folder named models exist
model_prefix = 'model1/chkpt'
checkpoint = mx.callback.do_checkpoint(model_prefix)
                                       
#Loading the module API. Previously mxnet used feedforward (deprecated)                                       
model =  mx.mod.Module(
    context = mx.gpu(0),     # use GPU 0 for training; if you don't have a gpu use mx.cpu()
    symbol = dense_1,
    data_names=['data']
   )
                                       
#actually fits the model for 3 epochs.                                    
model.fit(
    train_iter,
    eval_data=val_iter, 
    batch_end_callback = mx.callback.Speedometer(batch_size, 64),
    num_epoch = 3, 
    eval_metric='acc',
    optimizer = adam,
    epoch_end_callback=checkpoint
)
``` 


 ## Visualization
The following code will help us to visualize the result and can provide some intutiton about our model. We obtain the weights of embedding generated by our model. We then visualize the result using [tnse](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) visualization. Visualizing vocab-size*embedding_dim (5000*500) vector in 2-d is impossible. We need to reduce the dimensionality of the data inorder to visualize it. t-nse is popular visualization technique that can reduce the dimensionality  so that vector that are closer in N (500) dimensions are close together in lower dimension(2). It basically combines the N dimension vector into a reduced dimension with minimal loss of information. t-sne is very similar to [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis).

Below is the code for extracting the embedding weights and then visualizing as a scatter plot.

```python
# obtains the weights of embeding layer for visualizing
params = model.get_params()
print(params)
weights_dense_embed = model.get_params()[0]['vocab_embed_weight'].asnumpy()
weights_dense_embed = weights_dense_embed.astype('float64')

#tnse visualization for first 500 words
size = 500
Y= tsne(weights_dense_embed[:size])
plt.figure(0)
plt.scatter(Y[:, 0], Y[:, 1])
for idx, (x, y) in enumerate(zip(Y[:, 0], Y[:, 1])):
    plt.annotate(idx2word[idx], xy=(x, y), xytext=(0, 0), textcoords='offset points')
```

Here are the visualized traffic signs, with their labels:
![Alt text](images/vis.png?raw=true "traffic sign visualization")

The visualization shows thats the model automatically learnt that the words "excellent,wonderful,amazing" means the same thing. This is quite wonderful. Next we will look into convulution network


## Sample prediciton 

Lets us use this model to predict our sentences. For this we need to load the saved model as follows

```
# Load the model from the checkpoint , we are loading the 10 epoch
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)

# Assign the loaded parameters to the module
mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.bind(for_training=False, data_shapes=[('data', (1,500))])
mod.set_params(arg_params, aux_params)
```
Then encode the sentences with word_dict's index, and pass it to the model as classifier.

```
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

#a simple predict function
def predict(sen):
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(sen)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    return prob

#our custom sentences for testing
my_sen =["the movie was awesome. Loved it"]
my_sen_encoded = encoded_sentences(my_sen,word_dict)
my_sen_encoded_padded = pad_sequences(my_sen_encoded)
```

```
The output is 
[ 0.09290998  0.90709001] which means, the classifier predicts the sentiment is positive with 0.90 probabilty.
```

## building the convolution model.
As discussed earlier, to develop a model based on n-grams, we need convolution neural network. So let us develop  one. This is very similar to previous model, but has a convolution filter of size 5.

```python
#a model with convolution kernel of 5 (5-grams)
input_x_2 = mx.sym.Variable('data')
embed_layer_2 = mx.sym.Embedding(data=input_x_2, input_dim=vocab_size, output_dim=embedding_dim, name='vocab_embed')
conv_input_2 = mx.sym.Reshape(data=embed_layer_2, target_shape=(batch_size, 1, seq_len, embedding_dim))
conv1_2 = mx.sym.Convolution(data=conv_input_2, kernel=(5,embedding_dim), num_filter=100, name="conv1")
flatten_2 = mx.sym.Flatten(data=conv1_2)
fc2_2 = mx.sym.FullyConnected(data=flatten_2, num_hidden=2,name="final_fc")
convnet = mx.sym.SoftmaxOutput(data=fc2_2, name='softmax')
```

The only tricky thing is the 'conv_input_2' input layer. This layer reshapes the output from embedding layer into a format that is needed by convolution layer. Everything else remain the same. This model can trained using model.fit function and various insights can be obtained.

## building the mutiple convolution model.
This is similar to the previous model, except that we use conovultion of size of different size 3,4,5 for developing model based on 3-gram,4-gram,5-gram, then concatenates the output and then flatten the output. A maxpool layer is added to prevent overfitting. Below is the code in python

```
# a model with convolution filters of 3,4,5 (3-gram,4-grams,5-grams)
input_x_3= mx.sym.Variable('data')
embed_layer_3 = mx.sym.Embedding(data=input_x_3, input_dim=vocab_size, output_dim=embedding_dim, name='vocab_embed')
conv_input_3 = mx.sym.Reshape(data=embed_layer_3, target_shape=(batch_size, 1, seq_len, embedding_dim))


# create convolution + (max) pooling layer for each filter operation
filter_list=[3, 4, 5] # the size of filters to use

num_filter=100
pooled_outputs = []
for i, filter_size in enumerate(filter_list):
    convi = mx.sym.Convolution(data=conv_input_3, kernel=(filter_size, embedding_dim), num_filter=num_filter)
    relui = mx.sym.Activation(data=convi, act_type='relu')
    pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(seq_len - filter_size + 1, 1), stride=(1,1))
    pooled_outputs.append(pooli)

# combine all pooled outputs
total_filters = num_filter * len(filter_list)
concat = mx.sym.Concat(*pooled_outputs, dim=1)



# reshape for next layer
h_pool_3 = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))
fc2_3 = mx.sym.FullyConnected(data=h_pool_3, num_hidden=2,name="final_fc")
convnet_combined = mx.sym.SoftmaxOutput(data=fc2_3, name='softmax')
```


## Transfer learning from glove
The embedding we generated using neural network gives us lot of insights. But to generate a embedding, we need a lot of data. What if we are restricted to small amount of data? Transfering weights from different pre-trained neural network can be very useful. This helps to build model with small amount of data.

Here, we will use [glove](https://nlp.stanford.edu/projects/glove/) embedding developed by Stanford. We will use the wikipedia - 2014 glove embedding. This embedding was trained on  wikipedia corpus of 6 billions words. The embedding itself has 400k unique words(vocabulary)
and contains embedding with variousd dimension 50,100,200,300. We will use the 50 dimension embedding. The following functions loads the embedding into a embeddin matrix (numpy matrix)

```python
#loads glove word embedding 
def load_glove_index(loc):
    f = open(loc,encoding="utf8")
    embeddings_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

#creates word embedding matrix
def create_emb():
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_dict.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
embeddings_index = load_glove_index("glove/" + 'glove.6B.50d.txt')
embedding_matrix = create_emb();

```

Let us visualize the embedding of glove using t-nse.

```python
#visualization of word embedding.
size = 500
Y= tsne(embedding_matrix[:size])
plt.figure(1)
plt.scatter(Y[:, 0], Y[:, 1])
for idx, (x, y) in enumerate(zip(Y[:, 0], Y[:, 1])):
    plt.annotate(idx2word[idx], xy=(x, y), xytext=(0, 0), textcoords='offset points')
```

As you can see, the embedding has grouped similar words(worse,worst,bad,terrible) together. Let us use this embedding to create a neural network to classify sentences. Adding the pretrained weights to neural network can be little tricky. Below is the code

```
#creates a model with convolution and pretrained word embedding.
weight_matrix = mx.nd.array(embedding_matrix)
input_x_3= mx.sym.Variable('data')
the_emb_3 = mx.sym.Variable('weights') #the weight variable which will hold the pre trained embedding matrix
embed_layer_3 = mx.sym.Embedding(data=input_x_3,weight=the_emb_3, input_dim=vocab_size, output_dim=embedding_dim, name='vocab_embed')
conv_input_3 = mx.sym.Reshape(data=embed_layer_3, target_shape=(batch_size, 1, seq_len, embedding_dim))
conv1_3 = mx.sym.Convolution(data=conv_input_3, kernel=(5,embedding_dim), num_filter=100, name="conv1")
flatten_3 = mx.sym.Flatten(data=conv1_3)
fc2_3 = mx.sym.FullyConnected(data=flatten_3, num_hidden=2,name="final_fc")
convnet_word2vec = mx.sym.SoftmaxOutput(data=fc2_3, name='softmax')
```

Here, we first convert the glove embedding_matrix into a mx.nd.array. Then we create a symbol variables named weights and assign it to a  variable the_emb_3.  This variable is passed as a parameter to the embed_layer_3 in mx.sym.Embedding.

The next step is to train the neural network my passing the weight_matrix as the default weights for the embed_layer_3. Also, we need to freeze the weights of the embedding layer so during back propagation, the weights of embed_matrix is not updated. Remeber to pass fixed_param_names =['weights'] in the Module API to freeze the weights of embedding layer. Also,  pass the arg_params={'weights': weight_matrix} while fitting to use the glove embedding weights. Below is the python code.


```
adam = mx.optimizer.create('adam')

#Checkpointing (saving the model). Make sure there is folder named model4 exist
model_prefix_3 = 'model4/chkpt'
checkpoint = mx.callback.do_checkpoint(model_prefix_3)
                                       
#Loading the module API. Previously mxnet used feedforward (deprecated)                                       
model_3 =  mx.mod.Module(
    context = mx.gpu(0),     # use GPU 0 for training; if you don't have a gpu use mx.cpu()
    symbol = convnet_word2vec,
     fixed_param_names =['weights'] # makes the weights variable non trainable. Back propagration will not update #this variable
   )
                                       
#fits the model for 5 epochs.                                       
model_3.fit(
    train_iter,
    eval_data=val_iter, 
    batch_end_callback = mx.callback.Speedometer(batch_size, 64),
    num_epoch = 5, 
    eval_metric='acc',
    optimizer = adam,
    epoch_end_callback=checkpoint,
    arg_params={'weights': weight_matrix}, #loads the pretrained glove embedding to weights variable
    allow_missing= True
```








