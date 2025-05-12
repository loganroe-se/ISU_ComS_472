# Course Number
ComS 472 - Principles of Artificial Intelligence

# Class Overview
This class was primarily for learning the basics of artificial intelligence. The three projects were to teach some basics of different algorithms, all in Python, that can be used for searching, adversial searching, and adversial searching with random probabilities.

# Year Taken
Senior Year - Semester 2

# Grade Received
* Overall Class Grade: Unknown
* Projects:
  * Project I: A
  * Project II: A
      * Received **first place** in the class so +60% on final exam, averaging nearly 2.0 points.
  * Project III: Unknown

# Size of Group
Number of People: 2
* Myself, Logan Roe
* Zachary Foote

# Goal of the Projects
**Preface for ALL Projects**
* All projects are performed with a 30x30 grid where obstacles are marked with a 1 (i.e. a world is an MxM matrix such that world[0][0] == 1 if (0, 0) contains an obstacle, else, world[0][0] == 0 to indicate no obstacle). This world map is given to each project's algorithm immediately.
* Project I does not allow staying still, but Project II & III do. Outside of this, all projects allow 8 movements including orhtogonal & diagonal moves, all costing 1.

**Project I**  
The goal of this project was to create an algorithm that could navigate from a starting position to a goal position, while avoiding obstacles and staying within the bounds of the world, in the most efficient way possible. The world, as well as starting coordinates, and goal coordinates, are all given to the algorithm upfront. 

**Preface for Projects II & III**  
A specific setup is provided in terms of attacking/defending:
* Three players: Tom, Jerry, Spike
  * Tom's goal is caputre Jerry & evade Spike
  * Jerry's goal is capture Spike & evade Tom
  * Spike's goal is capture Tom & evade Jerry
* Scoring System:
  * If one player captures another, that player receives 3 points and the other 2 receive 0
      * I.e. if Tom captures Jerry, the scores will look like this: [T, J, S] --> [3, 0, 0]
  * If the iteration limit is hit, all players receive 1 point each
      * I.e. if no captures occur and the iteration limit is hit, the scores will look like this: [T, J, S] --> [1, 1, 1]
  * If a player runs into an obstacle or out of the map, they receive 0 points and all others receive 2 points
      * I.e. if a Tom runs into an obstacle before a capture or iteration limit occurs, the scores will look like this: [T, J, S] --> [0, 1, 1]
* Every player knows the entire world map (including obstacles) on each move. They also know the current location of the two opponents.
* The moves are made synchronously, so Tom does not know what Jerry is going to do. He may try to predict Jerry's move, but he does not know until it has already occurred.

_Class Placements_:
To determine class placement, all of the algorithms are run against each other and the top 8 teams moved to the playoffs. In the playoffs, the players competed until the professor had enough certainty that the results were solid and would not be swayed by more runs.
Extra credit placement for the final (does not stack from Project II and III, so max possible is 60% if you win):
* 1st Place - 60%
* 2nd Place - 40%
* 3rd Place - 30%
* All others in playoffs - 20%

**Project II**  
The goal of this project was to create an algorithm that could perform the following:
* Pursue & capture an opponent before captured yourself and do not run into any obstacles or go outside of the world's bounds.
* Be the team with the highest score at the end of the playoffs to get the most extra credit and win the game.

**Project III Twist**  
Project III is the exact same as Project II aside from one major thing: random probabilities affecting moves.  
Three probabilities existed: [P1, P2, P3] such that P1 + P2 + P3 = 1:
* P1 - Defined the probability that the desired move would be rotated 90 degrees counterclockwise.
* P2 - Defined the probability that the desired move would be executed as intended.
* P3 - Defined the probability that the desired move would be rotated 90 degrees clockwise.
Importantly, we were not given these probabilities at the start, we had to _calculate_ them ourselves based on intended moves and actual moves. With enough data, this could result in predicting modifications and trying to out-do the modification by using probability to your advantage. This also made it important to avoid nearby obstacles as you could be forced into one.

**Project III**  
The goal of this project was to create an algorithm that could perform the following:
* Pursue & capture an opponent before captured yourself and do not run into any obstacles or go outside of the world's bounds.
* Be the team with the highest score at the end of the playoffs to get the most extra credit and win the game.
* _Important Twist_: Calculate probabilities and base your moves off those probabilities to ensure accuracy & avoiding obstacles.

# Notes
We received first place on Project II and Project III placements are yet to be determined. In Project II, we scored nearly an average of 2.0 points, which is very good considering you can get a 0, 1, or 3 for each run. Though we did not get to see final scores for everyone, we were told that we beat the others out by a pretty decent margin, not within margin of error whatsoever.

To run Projects II and III yourself, you would need to populate tom.py, jerry.py, and spike.py with valid code. This could be as simple as a DFS, BFS, randomizer, etc. or it could be as complicated as a full A*, Minimax, Minimax with Alpha-Beta Pruning, MCTS, Q-Learning, etc. Feel free to see if you could out-perform our submissions!

**Disclaimer: Planner.py in each project's folder is the only code that we produced. All other code was provided by the professor, and we do not claim it as our own work. The report, however, is entirely our own work.
