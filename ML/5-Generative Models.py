5.1 Naive Bayes classifier 

# load the iris dataset
from sklearn.datasets import load_iris 
iris = load_iris()
# store the feature matrix (X) and response vector (y)
X = iris.data 
y = iris.target
# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 
# training the model on training set
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# making predictions on the testing set 
y_pred = gnb.predict(X_test)
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test,
                                                                           y_pred)*100)









5.2 Hidden Markov Models using hmmlearn

# import the necessary libraries
!pip install hmmlearn
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from hmmlearn import hmm
# Define the state space 
states = ["Sunny", "Rainy"] 
n_states = len(states)
print('Number of hidden states :',n_states) 
# Define the observation space 
observations = ["Dry", "Wet"] 
n_observations = len(observations)
print('Number of observations :',n_observations)


# Define the initial state distribution 
state_probability = np.array([0.6, 0.4]) 
print("State probability: ", state_probability) 
# Define the state transition probabilities 
transition_probability = np.array([[0.7, 0.3],
                                   [0.3, 0.7]])
print("\nTransition probability:\n", transition_probability) 
# Define the observation likelihoods 
emission_probability= np.array([[0.9, 0.1],
                                [0.2, 0.8]])
print("\nEmission probability:\n", emission_probability)


model = hmm.CategoricalHMM(n_components=n_states) 
model.startprob_ = state_probability
model.transmat_ = transition_probability 
model.emissionprob_ = emission_probability 
# Define the sequence of observations
observations_sequence = np.array([0, 1, 0, 1, 0, 0]).reshape(-1, 1) 
observations_sequence


# Predict the most likely sequence of hidden states 
hidden_states = model.predict(observations_sequence) 
print("Most likely hidden states:", hidden_states)


log_probability, hidden_states = model.decode(observations_sequence, 
lengths = len(observations_sequence),
algorithm ='viterbi' ) 
print('Log Probability :',log_probability)
print("Most likely hidden states:", hidden_states)


# Plot the results 
sns.set_style("whitegrid")
plt.plot(hidden_states, '-o', label="Hidden State") 
plt.xlabel('Time step')
plt.ylabel('Most Likely Hidden State') 
plt.title("Sunny or Rainy") 
plt.legend()
plt.show()
