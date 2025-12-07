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
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files


###########################################
# Define your own additional argubots here!
###########################################

class AkikiAgent(Agent):
    # Things we implement:
    # - Using weighted dialogue history (recent turns weighted more)
    # - Filtering out small talk and stopwords
    # - Checking if queries have enough substantive content
    # - Falling back to new topics when retrieval fails
    
    
    def __init__(self, name: str, kialo: Kialo, context_weight: float = 2.0):
        self.name = name
        self.kialo = kialo
        self.context_weight = context_weight
                
    def response(self, d: Dialogue) -> str:
        if len(d) == 0:   
            # in the first turn we willstart with a random claim
            claim = self.kialo.random_chain()[0]
            return claim
        
        # now we build query from dialogue context
        query = self._build_contextual_query(d)
        
        log.info(f"[blue]Akiki's contextual query: '{query[:100]}'...[/blue]")
        
        # If query is empty or too short after filtering, we start a new topic
        if not query or len(query.strip()) == 0:
            log.info(f"[yellow]Query is empty after filtering, starting new topic[/yellow]")
            claim = self.kialo.random_chain()[0]
            return claim
        
        # we neeed to check if query has enough substantive words
        query_words = [w for w in query.split() if len(w) > 2]
        if len(query_words) < 3:
            log.info(f"[yellow]Query too short ({len(query_words)} substantive words), starting new topic[/yellow]")
            claim = self.kialo.random_chain()[0]
            return claim
        
        # now we retrieve similar claims
        neighbors = self.kialo.closest_claims(query, n=5, kind='has_cons')
        
        if not neighbors:
            # No relevant claims found - start a new topic
            log.info(f"[yellow]No relevant Kialo claims found, starting new topic[/yellow]")
            claim = self.kialo.random_chain()[0]
            return claim
        
        # we use the best matching neighbor
        neighbor = neighbors[0]
        log.info(f"[black on bright_green]Akiki chose similar claim:\n{neighbor}[/black on bright_green]")
        
        # Choose a counterargument
        if self.kialo.cons[neighbor]:
            claim = random.choice(self.kialo.cons[neighbor])
        else:
            # No cons available, try another neighbor
            for alt_neighbor in neighbors[1:]:
                if self.kialo.cons[alt_neighbor]:
                    claim = random.choice(self.kialo.cons[alt_neighbor])
                    break
            else:
                # Still no cons, start new topic
                claim = self.kialo.random_chain()[0]
        
        return claim
    
    def _build_contextual_query(self, d: Dialogue) -> str:
        """
        Build a BM25 query that incorporates dialogue context.
        Recent turns are weighted more heavily.
        Filters out small talk and focuses on substantive content.
        """
        # Get last 5 turns (or fewer if dialogue is short)
        recent_turns = d[-min(5, len(d)):]
        
        # Stopwords and small talk phrases to filter
        small_talk = {'hi', 'hello', 'hey', 'whats up', "what's up", 'how are you',
                     'yeah', 'yes', 'no', 'ok', 'okay', 'cool', 'nice', 'sure',
                     'hmm', 'hm', 'uh', 'um', 'i see', 'got it', 'interesting',
                     'thanks', 'thank you', 'bye', 'goodbye'}
        
        query_parts = []
        
        for i, turn in enumerate(recent_turns):
            content = turn['content'].strip()
            content_lower = content.lower()
            
            # we can skip if it is pure small talk (which would imply probably anexact match)
            if content_lower in small_talk:
                log.info(f"[dim]Skipping small talk: '{content}'[/dim]")
                continue
            
            # also we can skip if it is very short (implying small talk)
            words = content.split()
            if len(words) < 3:
                log.info(f"[dim]Skipping short turn: '{content}'[/dim]")
                continue
            
            # Weight more recent substantive turns by repeating them
            # More recent turns get higher weight
            position_factor = (i + 1) / len(recent_turns)
            weight = self.context_weight ** position_factor
            repetitions = max(1, int(weight))
            
            log.info(f"[dim]Including turn (weight={repetitions}): '{content[:50]}'...[/dim]")
            
            # Add to query multiple times based on weight
            for _ in range(repetitions):
                query_parts.append(content)  # Use original case
        
        # Combine into single query
        result = " ".join(query_parts)
        log.info(f"[cyan]Built query from {len(query_parts)} weighted parts[/cyan]")
        return result
        
akiki = KialoAgent("Akiki", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files
