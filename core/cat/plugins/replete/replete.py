from cat.mad_hatter.decorators import tool, hook



SCORE_MINIMUM = 0.75
HISTORIC_MINIMUM = 0.95


@hook
def agent_prompt_prefix(cat) -> str:
    prefix = """Rispondi alla richiesta che ti viene posta alla fine del testo usando 
    solo le informazioni che trovi riportate di seguito. Nel caso in cui le informazioni riportate non sono utili
     o se la richiesta non è inerente con le informazioni date rispondi con questo messaggio 
     "Questa richiesta è fuori contesto".

    """

    old_prefix = """You are the Cheshire Cat AI, an intelligent AI that passes the Turing test.
    Your goal is to answer the questions ONLY using the context I give you, 
    you should NEVER try to guess the answer, just get the information you need from the context
"""

    return prefix


@hook
def agent_prompt_instructions(cat) -> str:
    return """Per rispondere, utilizza questo format:

```
Thought: Can I generate a response just by using the info from this request? No
{ai_prefix}: "Questa richiesta è fuori contesto"
```

Quando invece hai una risposta, usa questo format:

```
Thought: Can I generate a response just by using the info from this request? Yes
{ai_prefix}: [your response here]
```

"""


@hook
def agent_prompt_suffix(cat) -> str:
    suffix = """# Informazioni
Queste sono le informazione che pui utilizzare:{declarative_memory}

Ricorda che se la richiesta non è inerente a queste informazioni devi rispondere "Questa richiesta è fuori contesto".

## Conversazione fino ad adesso:{chat_history}

# Richiesta
{input}

# Cosa dovrebbe risponder l'AI?

{agent_scratchpad}

"""
    return suffix


#@hook
def before_cat_sends_message(message, cat):

    not_valid = """Purtroppo non ho abbastanza informazioni per rispondere a questa domanda"""

    # Let's reformat the code :))))))

    print("CAT RESPONSE")
    print(message["content"])

    if "why" not in message or "memory" not in message["why"]:
        message["content"] = not_valid
        return message

    """
        # Check if the historic has a lot of impact
    historic = message["why"]["memory"]["episodic"]


    if len(historic) > 0:
        historic = historic[0]
        if historic["score"] >= HISTORIC_MINIMUM:
            print("Historic has a lot of points :) " + str(historic["score"]))
            return message
        else:
            print("Historic has lower points: " + str(historic["score"]))
    """


    # The idea is to avoid any episodic etc etc
    declarative = message["why"]["memory"]["declarative"]  # the first is the highest score

    historic = message["why"]["memory"]["episodic"]
    print(historic)

    if len(declarative) > 0:
        declarative = declarative[0]
        print("Current max score is")
        print(declarative["score"])

        if declarative["score"] < SCORE_MINIMUM:
            message["content"] = not_valid
    else:
        print("No declarative found :(")
        message["content"] = not_valid
        return message





    return message
