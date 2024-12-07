"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `evaluate.py`.
We've included a few to get your started."""

import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent
from kialo import Kialo

# Use the same logger as agents.py, since argubots are agents;
# we split this file 
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim   


class WeightedKialoAgent(KialoAgent):
    """Akiki: A Kialo-based agent that considers dialogue history and includes earlier turns only if similarity is low."""
    
    def response(self, d: Dialogue) -> str:
        if len(d) == 0:
            # First turn: Start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            # Start with only the most recent turn in the query.
            most_recent_turn = d[-1]['content']
            neighbors = self.kialo.closest_claims(most_recent_turn, n=3, kind='has_cons')
            
            if not neighbors or self._is_similarity_low(most_recent_turn, neighbors[0]):
                # If similarity is low, include earlier turns in the query.
                query = self._build_weighted_query(d)
                neighbors = self.kialo.closest_claims(query, n=3, kind='has_cons')
            else:
                # Use the most recent turn as the query.
                query = most_recent_turn
            
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim

    def _is_similarity_low(self, recent_turn: str, closest_claim: str) -> bool:
        """
        Check if the similarity between the most recent turn and the closest claim is low.
        """
        # Simulate BM25 similarity using token overlap (or use a metric if BM25 exposes scores)
        recent_tokens = set(self.kialo.tokenizer(recent_turn))
        claim_tokens = set(self.kialo.tokenizer(closest_claim))
        overlap = len(recent_tokens & claim_tokens) / len(recent_tokens | claim_tokens)
        threshold = 0.3  # Define a threshold for low similarity
        return overlap < threshold

    def _build_weighted_query(self, d: Dialogue) -> str:
        """
        Build a weighted query by including earlier turns in the dialogue.
        """
        query_parts = []
        weight = 1.0  # Start with full weight for the most recent turn
        
        for turn in reversed(d):  # Start with the most recent turn
            if turn['speaker'] != self.name:  # Focus on the user's turns
                query_parts.append(f"{turn['content']} * {weight:.2f}")
                weight *= 0.5  # Reduce weight for older turns
        
        query = " ".join(query_parts)
        log.info(f"[black on bright_blue]Constructed weighted query:\n{query}[/black on bright_blue]")
        return query    


    
# Akiko doesn't use an LLM, but looks up an argument in a database.  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files

# Instantiate Akiki with the same Kialo database as Akiko
akiki = WeightedKialoAgent("Akiki", Kialo(glob.glob("data/*.txt")))
###########################################
# Define your own additional argubots here!
###########################################
