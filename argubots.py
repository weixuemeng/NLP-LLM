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
        
akiki = AkikiAgent(
    name="Akiki",
    kialo=Kialo(glob.glob("data/*.txt")),
    context_weight=2.0
)

class RAGAgent(LLMAgent):
    """
    Aragorn: Retrieval-Augmented Generation agent.
    Combines:
      - LLM (like Alice) for paraphrasing + final response
      - Kialo retrieval (like Akiki) for structured arguments
    """

    def __init__(self, name: str, client, model: str, kialo: Kialo,
                **kwargs_llm):
        super().__init__(name=name, client=client, model=model,
                          **kwargs_llm)
        self.kialo = kialo

    # Step 1:  Ask the LLM what claim should be responded to. 
    def _paraphrase_last_turn_as_claim(self, d: Dialogue) -> str:
        """
        Ask the LLM: given this whole dialogue, what is the human's
        last turn really *claiming* or *implying*?
        Return a single explicit sentence or short paragraph.
        """
        dialogue_text = d.script()
        previous_turn = d[-1]['content'].strip()  # previous turn from user

        system_msg = (
            "You are an argument analyst. I gave this whole dialogue to you, you must rewrite "
            "the human's LAST TURN as an explicit claim.\n"
            "Your answer should:\n"
            "  - Your paraphrase should makes an explicit claim and can be better understood without the context\n"
            "  - Your answer show with a much longer statement with many more word types.\n"
            "Return ONLY the rewritten claim."
        )

        user_msg = (
            "Here is the dialogue:\n"
            "-------------------------------\n"
            f"{dialogue_text}\n"
            "-------------------------------\n\n"
            f"The human's last turn was:\n\"{previous_turn}\"\n\n"
            "Please rewrite that last human turn as an explicit claim:"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.kwargs_llm,
        )

        paraphrased = resp.choices[0].message.content.strip()
        log.info(f"[green]Paraphrased human claim: {paraphrased}[/green]")
        return paraphrased

    def _kialo_document(self, llm_claim: str, n_neighbors: int = 3) -> str:
        """
        Look up claims in Kialo that are similar to the explicit
        claim.  Create a short "document" that describes some of those claims and
        their neighbors on Kialo.
        """
        neighbors = self.kialo.closest_claims(llm_claim, n=n_neighbors, kind="has_cons")
        assert neighbors, "No claims to choose from; is Kialo data structure empty?"

        lines = []
        lines.append("Below are some possibly related claims and arguments from the Kialo debate website.\n")

        for i, c in enumerate(neighbors, start=1):
            lines.append(f"Claim {i}: \"{c}\"")

            if self.kialo.pros[c]:
                lines.append("Some arguments from other Kialo users in favor of that claim::")
                for pro in self.kialo.pros[c]:
                    lines.append(f"    * {pro}")

            if self.kialo.cons[c]:
                lines.append("Some arguments from other Kialo users against that claim::")
                for con in self.kialo.cons[c]:
                    lines.append(f"    * {con}")

            lines.append("") 

        doc = "\n".join(lines).strip()
        log.info(f"[cyan]Built Kialo document:\n{doc[:300]}...[/cyan]")
        return doc

    def response(self, d: Dialogue, **kwargs) -> str:
        """
          1. Query formation step: LLM paraphrases human's last turn as a claim.
          2. Retrieval step: Retrieve Kialo arguments about that claim.
          3. Retrieval-augmented generation
        """
        if len(d) == 0: # no previous turns 
            return super().response(d, **kwargs)

        # step 1: Paraphrase last turn as explicit claim from LLM 
        explicit_claim = self._paraphrase_last_turn_as_claim(d)

        # 2. Retrieve related Kialo content
        kialo_doc = self._kialo_document(explicit_claim)

        # 3. Build final prompt for the response
        dialogue_text = d.script()

        system_msg = (
            "You are Aragorn, a thoughtful conversational agent. "
            "Respond directly to the human's viewpoint. "
            "You may use the evidence provided, but do not mention any retrieval process "
            "or where the arguments came from. Keep your word polite and reasoned."
        )

        user_msg = (
            "Here is the current dialogue:\n"
            f"{dialogue_text}\n\n"
            "Here is an explicit paraphrase of the human's last claim:\n"
            f"{explicit_claim}\n\n"
            "Here are some related pros and cons claims and arguments from the Kialo Database:\n"
            f"{kialo_doc}\n\n"
            "Now please write your next turn in the dialogue, responding to the human's last claim.\n"
            " Write in 2-4 sentences."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **(self.kwargs_llm | kwargs),
        )

        reply = resp.choices[0].message.content.strip()
        log.info(f"[magenta]Aragorn's response:\n{reply}[/magenta]")
        return reply

aragorn = RAGAgent(
    name="Aragorn",
    client=None, 
    model="gpt-4o-mini", 
    kialo=Kialo(glob.glob("data/*.txt")),
    temperature=0.7,
    max_tokens=500,
)

class Awsom(RAGAgent):
    """
    Awsom = Aragorn (RAG) + CoT planning
    """

    def _cot_plan_and_answer(self, d: Dialogue, explicit_claim: str, kialo_doc: str) -> str:
        dialogue_text = d.script()

        user_prompt = (
            "You are Awsom, an argubot trying to broaden the human's mind.\n\n"
            "Here is the dialogue so far:\n"
            f"{dialogue_text}\n\n"
            "From this, you have inferred that the human's last turn claim is:\n"
            f"  \"{explicit_claim}\"\n\n"
            "You also know some related pros and cons claims and arguments from the Kialo Database::\n"
            f"{kialo_doc}\n\n"
            "Now let's do and think some open questions:\n"
            "1. Think as 'Awsom (private thought)', and try to analyze the human's ideas, motivations, "
            ", personality. What are their intention and how best to respond to help the human express? Plan a way "
            "to open their mind while staying respectful. \n"
            "2. Then, as 'Awsom', speak ONE short reply (1-2 sentences) that responds "
            "to the human and gently challenges or broadens their view.\n\n"
            "Format your output EXACTLY like this:\n"
            "Awsom (private thought): <your analyze and plan>\n"
            "Awsom: <your final reponse>\n"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Awsom. You need to think privately before you reply to the human. "
                    "Your private thoughts are labeled 'Awsom (private thought)'. "
                    "The line starting with 'Awsom (to <human>):' is the final response to the human."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.kwargs_llm,
        )

        full_text = resp.choices[0].message.content.strip()
        log.info(f"[cyan]Awsom CoT output:\n{full_text}[/cyan]")

        import re
        m = re.search(r"Awsom:\s*(.*)", full_text, re.DOTALL)
        if m:
            answer = m.group(1).strip()
            return answer
        else:
            return full_text

    def response(self, d: Dialogue, **kwargs) -> str:
        if len(d) == 0:
            return super(LLMAgent, self).response(d, **kwargs)  # or define a custom opener

        # same as RAG 
        explicit_claim = self._paraphrase_last_turn_as_claim(d)
        kialo_doc = self._kialo_document(explicit_claim)

        # with COT 
        return self._cot_plan_and_answer(d, explicit_claim, kialo_doc)
    
awsom = Awsom(
    name="Awsom",
    client=None,
    model="gpt-4o-mini",
    kialo=Kialo(glob.glob("data/*.txt")),
    temperature=0.7,
    max_tokens=500,
)
