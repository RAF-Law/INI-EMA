# UofG Level-4 Individual Project
Author: Lawrence Zhang
Supervisor: Jessica Enright

Project Title: Generating a population of sample players for an activation game
Project Description: The activation game is played on a grid occupied by farmers, knights and kings. One farmer begins enchanted and observed, and at each turn an enchanted character either senses, revealing all characters within its range, or enchants one observed character within that range; enchantment is transient, so activating a character breaks the existing chain. The objective is to observe all kings in as few turns as possible. An implementation is publicly available at https://github.com/DavidSLeslie/INI-EMA, together with a Gymnasium environment, a reinforcement learning agent, an oracle giving optimal solutions on small instances, and four hand-written strategies. What the codebase does not have is a population of players: the strategies present were written individually, they are few, and they do not cover the space of behaviour a learning agent should be tested against. Hand-writing more does not scale and biases the population towards strategies their author found natural.

The student will build a generator producing diverse sample players automatically, conforming to the existing strategy interface so that generated players are usable within the repository as it stands. One approach is to parameterise the existing linear-features strategy and search using behavioural descriptors, for instance mean chain length, the ratio of sensing to enchanting, and the extent to which knights are preferred to farmers, so that the archive is populated with players that differ in how they play rather than only in how well. Reinforcement learning agents trained under varied budgets provide a second source of players. Suggested evaluation would report coverage of the behaviour space, the spread of solve times relative to the oracle, stability of the population across grid sizes and compositions, and whether the generated players expose failures in the existing agent that the hand-written strategies do not.

The project requires Python and an interest in game search, graph searching, and possibly either reinforcement learning or evolutionary search.

Background reading. The repository itself is the most important reading.

## Meeting Minutes

### 24/9/2026
- First project meeting, asking about future considerations, why I chose this project, meeting minutes etc.
- Try with some simple strategies first.
- Considering map generation etc.
- Possible related fields:
 - graph searching
 - strategic searching
 - comparing strategies
 - heuristics
 - also check with the graphs/representations they use
- Read through repository √
- Configure the environment, run locally √
- Play with the game √

### 08/10/2026