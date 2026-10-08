"""
Code to support the EMA workshop at the INI in January 2026
This encodes the Activation Game posed by a workshop sponsor,
in a Gymnasium style environment.

David S Leslie
d.leslie@lancaster.ac.uk
"""

import numpy as np
import random
import gymnasium as gym

import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ActivationGameCharacter:
    """
    Class representing a character in the game
    Attributes are:
    - chartype: the type of the character (valid types stored in allowed_types)
    - range: the range of detection/influence of the character, determined by type
    - location: a list of length 2 giving the location in the world
    - isEnchanted: a Boolean indicating whether the character is currently enchanted
    - isEnchanting: a GameCharacter which this GameCharacter is enchanting
    - couldEnchant: a set of GameCharacters that are in range and have been observed
    - isObserved: a Boolean indicating whether this character has been observed yet
    Methods are:
    - enchant: set the agent to enchanted
    - disenchant: remove the enchanted status of this agent and all recursively enchanted agents
    - inRange: indicate whether a location is in the observable range of this character
    - step: take an action (sense or enchant a particular character)

    """
    def __init__(self, chartype, location=(-1,-1)):
        
        # Check and stoee the character's type
        allowed_types = ["Farmer","Knight","King"]
        if chartype not in allowed_types:
            raise ValueError(f"chartype is {chartype} but must be one of {allowed_types}")
        self.chartype = chartype
        # Allocate the range based on the type
        if chartype == "Farmer":
            self.range = 3
        elif chartype == "Knight":
            self.range = 4
        else:
            self.range = 0
        
        # Check and store the character's location
        try:
            x,y = location
        except:
            raise ValueError(f"location is {location} but needs to be a tuple with two entries")
        self.location = [x,y]

        # Characters start non-enchanted and non-observed
        self.isEnchanted = False
        self.isEnchanting = None
        self.couldEnchant = set()
        self.isObserved = False


    def enchant(self):
        """
        Method to set the enchanted status of the character
        """
        self.isEnchanted = True

    def disenchant(self):
        """
        Method to recursively disenchant all "children"
        """
        if self.chartype != "King":
            self.isEnchanted = False
        if self.isEnchanting is not None:
            # self.isEnchanting is the GameCharacter that we are currently emchanting
            # This character automatically needs to be disenchanted too
            self.isEnchanting.disenchant()
            self.isEnchanting = None
    
    def observe(self,world):
        """
        Method to take actions when the character is observed
        """
        if self.chartype != "King":
            # Kings do not get enchanted now, only observed.
            # May want to reverse this change at some point.
            for char in world.characters:
                if char.inRange(self.location) and char != self:
                    char.couldEnchant.add(self)
        self.isObserved = True
        
    
    def inRange(self,target_location):
        """
        Docstring for inRange
        
        :param target_location: tuple with two entries, location of a putative target

        Returns True if the target_location is in the sensing range, False otherwise
        """
        try:
            x, y = target_location
        except:
            raise ValueError(f"target_location is {target_location} but needs to be a tuple with two entries")
        
        if (abs(x-self.location[0])<=self.range) and (abs(y-self.location[1])<=self.range):
            return True
        else:
            return False


    def step(self,action,world):
        """
        Method for acting int the world
        
        :param action: the action to take. Either "sense" or a character to enchant
        :param world: the list of characters in the world

        Sets self.couldEnchant to be the characters that this agent could choose to enchant
        """

        if self.isEnchanted == False:
            raise RuntimeError(f"Character {self.location} is not enchanted but has been asked to perform an action")
        if self.chartype not in ["Farmer","Knight"]:
            raise ValueError("Only Farmers and Knights can act in the world")
        
        # If asked to take an action, any existing enchantments are broken
        if self.isEnchanted:
            self.disenchant()
            self.enchant()
 
        if action == "Sense":
            # Sensing means that we observe all the in-range characters
            for char in world.characters:
                if self.inRange(char.location):
                    char.observe(world)
            # Need to change the obs_mask of the world too, so that we know which locations are known to be empty
            # Just using a loop here. There are probably better ways, but I can't be bothered to be clever about
            # handling the edges of the world
            for ii in range(max(self.location[0]-self.range,0),min(self.location[0]+self.range+1,world.gridheight)):
                for jj in range(max(self.location[1]-self.range,0),min(self.location[1]+self.range+1,world.gridwidth)):
                    world.obs_mask[ii,jj] = True
        elif isinstance(action,ActivationGameCharacter):
            # if the action is a suitable character in the game, enchant that character
            if action not in self.couldEnchant:
                raise ValueError(f"Enchantment target at {action.location} is not enchantable right now")
            action.enchant()
            self.isEnchanting = action
        else:
            raise ValueError(f"Character action is {action} but must either be 'Sense' or a character to enchant")


class ActivationGameWorld:
    """
    A game world for the activation game
    """
    def __init__(self,gridsize = 20, composition=[10,10,4], seed = None, silent=False):
        """
        Docstring for __init__
        
        :param gridsize: size of the square game grid
        :param composition: a list of integers with 
        """

        # Check and process inputs
        if isinstance(gridsize,int) and (gridsize > 0):
            self.gridwidth = gridsize
            self.gridheight = gridsize
        else:
            raise ValueError(f"gridsize ({gridsize}) miust be a positive integer")

        try:
            nfarmers, nknights, self.nkings = composition
        except:
            raise ValueError("composition must be a list/tuple of form [nfarmers,nknights,nkings]")
        if not isinstance(nfarmers,int) or not isinstance(nknights,int) or not isinstance(self.nkings,int):
            raise ValueError("The entries in composition must be integers")
        
        # Set the seed if it has been passed
        if seed is not None:
            random.seed(seed)

        # Sample a world until we get a solvable one
        self.sample_world(composition)
        ntries = 0
        while not self.is_solvable() and ntries < 100:
#            print("Resampling world - previous world was not solvable")
            ntries += 1
            self.sample_world(composition)
        if ntries == 100:
            raise RuntimeError("Could not sample a solvable world after 100 tries")
        elif ntries > 0 and silent==False:
            print(f"Sampled {ntries} worlds before finding a solvable world")


        # Declare the whole grid to be unobserved
        self.obs_mask = np.zeros((self.gridheight,self.gridwidth), dtype = np.bool)

        # Activate the first farmer
        self.characters[0].enchant()
        self.characters[0].observe(self)

        # Initiate the step counter to 0
        self.nsteps = 0

    def sample_world(self,composition):
        """
        Method to sample a new world configuration
        """
        # Sample a set of locations for the characters uniformly across the grid
        nfarmers, nknights, nkings = composition
        character_locations = [divmod(ii,self.gridwidth) 
                               for ii in random.sample(range(self.gridwidth*self.gridheight),
                                                       nfarmers+nknights+nkings)
                                ]
        
        # Allocate characters to these locations
        self.characters = [ActivationGameCharacter("Farmer",loc) for loc in character_locations[:nfarmers]]
        self.characters.extend(
            [ActivationGameCharacter("Knight",loc) for loc in character_locations[nfarmers:(nfarmers+nknights)]]
        )
        self.characters.extend(
            [ActivationGameCharacter("King",loc) for loc in character_locations[(nfarmers+nknights):]]
        )

    def is_solvable(self):
        """
        Method to check whether the current world is solvable
        Returns True if it is, False otherwise
        """

        to_expand = [self.characters[0]]
        could_be_enchanted = set([self.characters[0]])

        # Recursively extend the putative enchantment set
        while len(to_expand) > 0:
            current = to_expand.pop()
            could_be_enchanted.add(current)
            # Add all characters in range from the current char to the to_expand list
            # if they have not already been expanded from
            for char in self.characters:
                if current.inRange(char.location) and char not in could_be_enchanted:
                    if char.chartype != "King":
                        to_expand.append(char)
                    else:
                        could_be_enchanted.add(char)
            
        # Now see if all kings are in the enchanted set
        solvable = all([char in could_be_enchanted for char in self.characters if char.chartype=="King"])
        return solvable


    def get_actions(self):
        """
        This method returns all the valid actions in the world
        The return value is a list of locations of the initiating agent, and either "Sense" or the location of an enchantable agent
        The reason for the format is so that we can use it reasonably in RL algorithms
        
        We assume for now that an already enchanted agent cannot be enchanted again by a different enchantress,
        but this is an interesting proposition - could we pass off the enchantment to someone else?（？）
        """

        sense_actions = [[char.location,"Sense"] for char in self.characters if char.isEnchanted and char.chartype != "King"]
        enchant_actions = [[enchanter.location,target.location] for enchanter in self.characters if enchanter.isEnchanted for target in enchanter.couldEnchant if not target.isEnchanted]
        
        actions = sense_actions + enchant_actions

        return actions
            
    
    def render(self,observer="Human"):
        """
        Docstring for gen_obs, method to generate observations for a player of the game
        
        :param observer:    if "SB3" then generate sb3-compatible observations
                            if "Human" then generate human-compatible observations
        """
        if observer == "Human" or observer == "Gamesmaster":
            # Start by noting all the observed grid
            if observer == "Human":
                grid = np.full((self.gridheight,self.gridwidth), fill_value='     ')
                for ii in range(self.gridheight):
                    for jj in range(self.gridwidth):
                        if self.obs_mask[ii][jj]:
                            grid[ii][jj] = '.    '
            else:
                grid = np.full((self.gridheight,self.gridwidth), fill_value='.    ')
            
            # Now insert the observed characters
            for char in self.characters:
                if char.isObserved or observer == "Gamesmaster":
                    if not char.isObserved:
                        enchanting_symbol = "x   "
                    elif char.isEnchanting is not None:
                        diff0 = char.isEnchanting.location[0]-char.location[0]
                        diff1 = char.isEnchanting.location[1]-char.location[1]
                        enchanting_symbol = f"{diff0:+2d}{diff1:+2d}"
                    else:
                        enchanting_symbol = '    '

                    if char.chartype == "Farmer":
                        typesymbol = 'F' if char.isEnchanted else 'f'
                    elif char.chartype == "Knight":
                        typesymbol = 'K' if char.isEnchanted else 'k'
                    else:
                        typesymbol = 'R' if char.isEnchanted else 'r'

                    symbol = typesymbol + enchanting_symbol
                    grid[char.location[0],char.location[1]] = symbol
        
            print(' ','   '.join([f"{ii:2d}" for ii in range(self.gridwidth)]))
            for ii in range(self.gridheight):
                print(f"{ii:2d}",''.join(grid[ii,:]))
            print("\n")
        elif observer == "sb3":
            raise NotImplementedError("Yet to implement the sb3 observation method")
        else:
            raise ValueError("observer must be in ['Human','Gamesmaster','SB3']")
        

    def step(self,action):
        """
        Method to take a step in the world
        
        :param action: the action to take, in the format returned by get_actions
        """

        try:
            initiator_loc, target = action
        except:
            raise ValueError(f"action is {action} but must be a list/tuple of form [initiator_location,target]")

        # Find the initiating character
        try:
            initiator = [char for char in self.characters if char.location == initiator_loc][0]
        except:
            raise ValueError(f"No character found at location {initiator_loc}")

        # Find the target character if needed
        if target != "Sense":
            try:
                target_char = [char for char in self.characters if char.location == target][0]
            except:
                raise ValueError(f"No character found at target location {target}")
            initiator.step(target_char,self)
        else:
            initiator.step("Sense",self)

        # Increment the step counter
        self.nsteps += 1


    def is_solved(self):
        """
        Method that checks whether all Kings have been observed
        (The criterion used to be all Kings were enchanted, but we
        realised that it never makes sense not to instantly enchant an observed King.
        It may be that some possibility of spontaneous disenchantment would make it
        more interesting / complex and introduce a value of actively deciding when to enchant.)
        """
        return(all([char.isObserved for char in self.characters if char.chartype=="King"]))
    
class ActivationGameEnv(gym.Env):
    """
    Gymnasium environment for the Activation Game
    """
    def __init__(self,gridsize=20,composition=[10,10,4],seed=None):
        super().__init__()

        self.world = ActivationGameWorld(gridsize=gridsize,composition=composition,seed=seed)

        self.gridsize = gridsize
        self.composition = composition
        self.max_obs_range = max([char.range for char in self.world.characters])

        self.action_space = gym.spaces.Discrete(gridsize**2 * (2*self.max_obs_range+1)**2)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4,gridsize,gridsize), dtype=np.uint8)

 
        ### Some parameters for reward shaping
        self.completion_reward = 1.0
        self.obs_reward = 0.2/(self.gridsize**2)
        self.king_activation_reward = 0.2/self.composition[-1]
        self.nonking_activation_reward = 0.1/(sum(self.composition[:-1]))
    
    def reset(self,seed=None):
        self.world = ActivationGameWorld(gridsize=self.gridsize,composition=self.composition,seed=seed)
        return self.__gen_obs(), {}
    
    def step(self,action):
        """
        Actions are encoded as integers

        Let num_cells be the number of cells in the grid 
        and num_action_cells be the maximal number of cells an agent can act upon

        The integer // num_action_cells is the actioning character's location (encoded as an integer)
        And integer % num_action_cells is the relative target location (encoded as an integer)

        Character location integers are loc[0]*gridwidth + loc[1]
        Relative target location integers are (rel_loc[0]*(2*max_obs_range+1) + rel_loc[1])+max_obs_range

        If relative target location == initiator location, then the action is "Sense"
        Otherwise the action is to enchant the character at the relative target location
        """
        # 

        # Count number of cells observed and kings enchanted before taking the step
        pre_observed = self.world.obs_mask.sum()
        pre_kings = sum([char.isEnchanted for char in self.world.characters if char.chartype=="King"])
        pre_nonkings = sum([char.isEnchanted for char in self.world.characters if char.chartype!="King"])


        num_grid_cells = self.world.gridwidth*self.world.gridheight
        num_action_cells = (2*self.max_obs_range+1)**2
        initiator, rel_target = divmod(action,num_action_cells)
        initiator = list(divmod(initiator,self.world.gridwidth))
        target = (np.array(initiator)+np.array(divmod(rel_target,2*self.max_obs_range+1))-self.max_obs_range).tolist()
        if target==initiator:
            world_action = [initiator,"Sense"]
        else:
            world_action = [initiator,target]

        # Take the action
        self.world.step(world_action)

        obs = self.__gen_obs()

        terminated = self.world.is_solved()
    
        reward = self.completion_reward * terminated
        reward += self.obs_reward * (self.world.obs_mask.sum()-pre_observed)
        reward += self.king_activation_reward * (
                sum([char.isEnchanted for char in self.world.characters if char.chartype=="King"])
                - pre_kings
        )
        reward += self.nonking_activation_reward * (
                sum([char.isEnchanted for char in self.world.characters if char.chartype!="King"])
                - pre_nonkings
        )

        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info
    
    def render(self):
        self.world.render(observer="Human")
    
 
    def close(self):
        pass  

    def action_masks(self) -> list[bool]:
        """
        Actions are encoded as integers

        Let num_cells be the number of cells in the grid 
        and num_action_cells be the maximal number of cells an agent can act upon

        The integer // num_action_cells is the actioning character's location (encoded as an integer)
        And integer % num_action_cells is the relative target location (encoded as an integer)

        Character location integers are loc[0]*gridwidth + loc[1]
        Relative target location integers are (rel_loc[0]*(2*max_obs_range+1) + rel_loc[1])+max_obs_range

        If relative target location == initiator location, then the action is "Sense"
        Otherwise the action is to enchant the character at the relative target location
        """
        valid_actions = self.world.get_actions()

        num_grid_cells = self.world.gridheight*self.world.gridwidth
        num_obs_cells = (2*self.max_obs_range+1)**2

        action_mask = [False]*(num_grid_cells*num_obs_cells)

        for (initiator,target) in valid_actions:
            if target=="Sense":
                target = initiator
            shifted_rel_pos = (np.array(target)-np.array(initiator)+ self.max_obs_range).tolist()
            initiator_index = initiator[0]*self.world.gridwidth + initiator[1]
            target_index = shifted_rel_pos[0]*(2*self.max_obs_range+1) + shifted_rel_pos[1]
            action_index = initiator_index*num_obs_cells + target_index
            action_mask[action_index] = True

        return action_mask

    def __gen_obs(self):
        """
        The machine learning ready observation of the world
        We present observations as a multichannel 'image'
        Channel 0 reports the observation mask
        Channel 1 reports the observed characters, including whether they are enchanted
            (1 is farmer, 2 is knight, 3 is king, add 16 if they are enchanted)
        Channel 2 and 3 report the location of the characters that are enchanted
        """

        # Obs mask into channel 0
        channel0 = self.world.obs_mask.astype(np.uint8)

        # Observed characters into channel 1
        channel1 = np.zeros((self.world.gridheight,self.world.gridwidth), dtype=np.uint8)
        for char in self.world.characters:
            if char.isObserved:
                loc = char.location
                if char.chartype == "Farmer":
                    base = 1
                elif char.chartype == "Knight":
                    base = 2
                else:
                    base = 3
                if char.isEnchanted:
                    base += 16
                channel1[loc[0],loc[1]] = base
        
        # Locations of enchanted characters into channels 2 and 3
        # Initialise to -1 (mod size of np.uint8) so that unenchanting is different from enchanting [0,0]
        channel2 = np.zeros((self.world.gridheight,self.world.gridwidth), dtype=np.uint8) - 1
        channel3 = np.zeros((self.world.gridheight,self.world.gridwidth), dtype=np.uint8) - 1
        for char in self.world.characters:
            if char.isEnchanting is not None:
                enchantress = char.location
                enchanted = char.isEnchanting.location
                channel2[enchantress[0],enchantress[1]] = enchanted[0]
                channel3[enchantress[0],enchantress[1]] = enchanted[1]

        return np.stack([channel0,channel1,channel2,channel3], axis=0)
    



class ActivationGameCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=7),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))