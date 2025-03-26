# User simulation using ideas from computational rationality for an extremely simplified search task.
The task of searching for an optimal web page can be formulated as a Partially Observable Markov Decision Process. That way, we can simulate the user behavior by using Reinforcement Learning to solve this POMDP and study the emergent behavior.

## Task
Navigate web pages through links to find an optimal webpage.
The web pages are structured into a binary tree (every web-page leads to 2 more web-pages up to some depth).
The value of the first web page is 50.
Every subsequent webpage has some return sampled from the normal distribution centered at its parent's value with a standard deviation of 10.

## User limitation
The user perceives the real return values of a page noisy.
The users perception is limited in their cognitive ability.
The cognitive ability affects:
- the initial noise level of the pages perception
- the rate of decay of the noise (spending more time on a page lets them perceive the real value more clearly)

## Observation, Action, Reward function
At each timesteps the **observation** of the user is:
- the current nodes' perceived value
- the first childs' perceived value
- the second chils' perceived value

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

Run
```
python3 main.py
```
to run experiments in several settings. The output will be logged in _results.csv_.