import torch

X, y = torch.load('somethingdata')

for epoch in range(max_epochs):
    for i in range(n_batches):

        local_X, local_Y = something something 

        # your model from here 



# a good way to keep track of samples and their labels is to adopt the following framework 

# -> create a dictionary called partition where you gather:

#   : in partition['train'] a list of training Id's 
#   : in partition['validation'] a list of validation ID's 

# -> create a dictionary called labels where for each ID of the dataset, the associated label is given by labels[ID]








# Each call requests a sample index for which the upperbound is specified in the __len__ method.

