# Simulating users in a simple search task
We made a user simulation using ideas from computational rationality for an extremely simplified search task.
The task of searching for an optimal web page can be formulated as a Partially Observable Markov Decision Process. That way, we can simulate the user behavior by using Reinforcement Learning to solve this POMDP and study the emergent behavior.

## Task
Navigate web pages through links to find an optimal webpage.
The web pages are structured into a binary tree (every web-page leads to 2 more web-pages up to some depth). Binary search is of course already solved optimally in computer science, but it is interesting how a cognitively constrained human (constrained by their memory, perception, and patience) is doing in  this task.
The value of the first web page is 50.
Every subsequent webpage has some return sampled from the normal distribution centered at its parent's value with a standard deviation of 10.

## User limitation
The user perceives the real return values of a page noisily.
The users' perception is limited in their cognitive ability.
The cognitive ability affects:
- the initial noise level in the user's perception of a pages' value
- the rate of decay of the noise (spending more time on a page lets the user perceive the real value more clearly)

## Observation, Action, Reward function
At each timesteps the **observation** of the user is:
- the current nodes' perceived value
- the first childs' perceived value
- the second childs' perceived value

The **action** a user can perform is:
- go back to the home page (root node)
- navigate back one page (parent node of current node)
- navigate to the first link (first child)
- navigate to the second link (second child)
- stay on the current page and study it
- choose the current node (ends episode)

The **reward** given to the user is sparse.
- they receive a penalty every timestep corresponding to their patience level
- at the end of the episode (either if they choose to end the search or if they maximum amount of steps is reached) they get the true value of the current page as a reward


## Instructions for running the code

### Installation
First install python3 and pip. Tested on Python 3.12.3 and pip 24.0
Then execute:
```
python3 -m venv path/to/virtual/environment
source path/to/virtual/environment/bin/activate
pip install -r requirements.txt
```

### Run experiment
Execute the command
```
python3 main.py
```
to run experiments in several settings. The output will be logged in _results.csv_.

## Initial results
The results suggest that with lower cognitive ability the final return that the user achieves in the search task decreases and the time they take increases. However, with less patience the user likes to spend less time on the search. Therefore, also the patience decreases the final return. If low cognitive ability is coupled with little patience, the worst performances are achieved.
More patience can make up for lower cognitive ability, there is a trade-off between the two. E.g. a user A who shows twice as much patience as user B can perform better on average even if user B has 1.5 times faster perception (reading) speed.


## Future steps
In the future this repository should be expanded:
1. from simple binary search to more complex search with multiple links per page and loops in the linkings
2. nodes with actual content (e.g. wikipedia pages)
3. search by keywords (LLM powered search)
4. visual POMDP: search by navigation through web pages
5. validation of user model through user study