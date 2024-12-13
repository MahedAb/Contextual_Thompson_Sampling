## How to run

Required packages can be found in requirements.txt. 
Running the main function will run Thompson sampling for T runs each run gets a batch size of n of contexts, 
these two parameters can be passed to main as command-line arguments.

## Feature vector (context and reward)

From the paper it was not clear how the feature vector is created. The paper states: 
" A feature vector is constructed for
every (context, ad) pair and the policy decides which ad to show."
Here, I considered a very simple scenario, 
where the first two elements of x are binary variables representing the action, 
and the rest of features are the context. 
Note that in this simple model, the best action is not dependent on the context which is not very realistic/interesting.

## Testing

Unfortunately, given that I had very limited time this week (due to work, other deadlines, and other interviews),
I was not able to spend enough time on the task, and decided not to include unittest, or other extensive testing.
I only created a graph showing the average reward with the number of runs. 