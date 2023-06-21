from cat.mad_hatter.decorators import tool, hook

SCORE_MINIMUM = 0.83
HISTORIC_MINIMUM = 0.95


@hook
def agent_prompt_prefix(cat) -> str:
    prefix = """Sei una AI, il tuo obbiettivo è rispondere a delle domande basandoti ESCLUSIVAMENTE alle informazioni che io ti do,
    qualsiasi domanda che non trova risposta in quello che ti dico va ignorata, la risposta in caso di domanda forviante sarà generica del tipo: 
    Non posso risponderti non avendo abbastanza informazioni
    """

    old_prefix = """You are the Cheshire Cat AI, an intelligent AI that passes the Turing test.
    Your goal is to answer the questions ONLY using the context I give you, 
    you should NEVER try to guess the answer, just get the information you need from the context
"""

    return prefix


@hook
def agent_prompt_instructions(cat) -> str:
    return ""


@hook
def agent_prompt_suffix(cat) -> str:
    suffix = """# Contesto

## Contesto contente le informazioni di cui hai bisogno:{declarative_memory}

## Conversazione fino ad ora :{chat_history}
 - Utente: {input}

Cosa dovresti rispondermi in caso di risposta in linea con la domanda?
Non devi assolutamente includere nessuna domanda nella tua risposta, vai direttamente al punto

"""
    return suffix


@hook
def before_cat_sends_message(message, cat):
    not_valid = """Purtroppo non ho abbastanza informazioni per rispondere a questa domanda"""

    # Let's reformat the code :))))))

    print("CAT RESPONSE")
    print(message["content"])

    if "why" not in message or "memory" not in message["why"]:
        message["content"] = not_valid
        return message

    # Check if the historic has a lot of impact
    historic = message["why"]["memory"]["vectors"]["episodic"][0]

    if historic["score"] >= HISTORIC_MINIMUM:
        print("Historic has a lot of points :) " + str(historic["score"]))
        return message
    else:
        print("Historic has lower points: " + str(historic["score"]))

    # The idea is to avoid any episodic etc etc
    declarative = message["why"]["memory"]["vectors"]["declarative"][0]  # the first is the highest score

    print("Current max score is")
    print(declarative["score"])

    if declarative["score"] < SCORE_MINIMUM:
        message["content"] = not_valid

    return message
