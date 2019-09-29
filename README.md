Objective

To design a Multilayer perceptron that can classify spoken words from (0-9). We can use Kaggle speech dataset for training. Ideally, the classifier should have high accuracy with minimal MLP architecture that possibly can be developed.

Exploration of Input Data

The input files are audio wav files for each of the spoken words with different retries by a set of speakers. To avoid overfitting, a percentage of the spoken words have to be kept aside for testing, and the data should not be picked up by the training samples. Also in order to avoid overfitting, spoken words from certain speaker is always mapped to test data only. We can maintain the same split by following the classification already done in the dataset.

Not all the files are of the same time length and there are shorter clips as well. So in order to make use of all the training set, data should be clipped/ padded for maximized use. Also audio clips can be seen having silence part as well, which can be clipped to reduce dimension.

Data Preprocessing

Raw audio file is too random signal to process and detect a pattern. So audio file must be transformed to frequency domain and in our case , we have tried doing a simple spectrogram as well as in logarithmic scale. Logarithmic scale tends to be better choice for preprocessing to contain all the useful frequencies distributed evenly and also we can use standard scalars available with scikit toolset to fit the range of input data between zero to one before feeding to MLP NN.Also target labels have to be one hot encoded for fair comparison for classification.

Baseline Model

NN with one hidden layer and one output layer with 10 neurons since no of classes = 10. Number of neurons at hidden layer is selected mid way between number of inputs and number of output. Relu activation for hidden layer and Softmax for output layer. Optimizer can be popular ‘adam’ with default learning rate parameter. Standard batch size of 32 and 10 Epochs Training data to be shuffled at end of every epoch. categorical_crossentropy as loss function due to balanced classification problem. Kernels initialized with ‘he_normal’ for good initial start of weights. Dropout layer added to get effect of ensemble of multiple neural networks trained at same time.

Future Considerations

Compare with data augmentation ( add padded samples) Dimension reduction ( clip and make shorter time durations, resample data) Run for longer epoch in a GPU compare against CNN, and AutoEncoder and any pertained networks tuned towards audio recognition Explore border Pairs method where instead of Grid / Random search for MLP’s exploding hyperparameters, arrive at minimal MLP architecture directly.

References

https://wikidelia.net/wiki/Spectrograms#Logarithmic_frequency_axis https://towardsdatascience.com/speech-classification-using-neural-networks-the-basics-e5b08d6928b7 https://www.kaggle.com/davids1992/speech-representation-and-data-exploration https://web.archive.org/web/20140721050413/http://www.heatonresearch.com/node/707
