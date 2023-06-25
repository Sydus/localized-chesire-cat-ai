from cat.mad_hatter.decorators import tool, hook
from cat.log import log

SCORE_MINIMUM = 0.83
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
    return """"""


@hook
def agent_prompt_suffix(cat) -> str:
    suffix = """# Informazioni
Queste sono le informazione che pui utilizzare:{declarative_memory}

Ricorda che se la richiesta non è inerente a queste informazioni devi rispondere "Questa richiesta è fuori contesto".

# Richiesta
{input}

"""
    return suffix


# @hook
def before_cat_sends_message(message, cat):
    not_valid = """Purtroppo non ho abbastanza informazioni per rispondere a questa domanda"""

    # Let's reformat the code :))))))

    log("CAT RESPONSE")
    log(message["content"])

    if "why" not in message or "memory" not in message["why"]:
        message["content"] = not_valid
        return message

    # Check if the historic has a lot of impact
    historic = message["why"]["memory"]["episodic"]

    if len(historic) > 0:
        historic = historic[0]
        if historic["score"] >= HISTORIC_MINIMUM:
            log("Historic has a lot of points :) " + str(historic["score"]))
            return message
        else:
            log("Historic has lower points: " + str(historic["score"]))

    # The idea is to avoid any episodic etc etc
    declarative = message["why"]["memory"]["declarative"]  # the first is the highest score

    if len(declarative) > 0:
        declarative = declarative[0]
        log("Current max score is")
        log(declarative["score"])

        if declarative["score"] < SCORE_MINIMUM:
            message["content"] = not_valid
    else:
        log("No declarative found :(")
        message["content"] = not_valid
        return message

    return message
