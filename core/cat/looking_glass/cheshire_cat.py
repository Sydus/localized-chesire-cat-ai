import time
from copy import deepcopy
import traceback

import langchain
import os
from cat.log import log
from cat.db.database import get_db_session, create_db_and_tables
from cat.rabbit_hole import RabbitHole
from cat.mad_hatter.mad_hatter import MadHatter
from cat.memory.working_memory import WorkingMemory
from cat.memory.long_term_memory import LongTermMemory
from cat.looking_glass.agent_manager import AgentManager
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate


# main class
class CheshireCat:
    def __init__(self):
        # access to DB
        self.load_db()

        # bootstrap the cat!
        self.bootstrap()

        # queue of cat messages not directly related to last user input
        # i.e. finished uploading a file
        self.web_socket_notifications = []

    def bootstrap(self):
        """This method is called when the cat is instantiated and
        has to be called whenever LLM, embedder,
        agent or memory need to be reinstantiated
        (for example an LLM change at runtime)
        """

        # reinstantiate MadHatter (reloads all plugins' hooks and tools)
        self.load_plugins()

        # allows plugins to do something before cat components are loaded
        self.mad_hatter.execute_hook("before_cat_bootstrap")

        # load LLM and embedder
        self.load_natural_language()

        # Load memories (vector collections and working_memory)
        self.load_memory()

        # After memory is loaded, we can get/create tools embeddings
        self.mad_hatter.embed_tools()

        # Agent manager instance (for reasoning)
        self.agent_manager = AgentManager(self)

        # Rabbit Hole Instance
        self.rabbit_hole = RabbitHole(self)

        # allows plugins to do something after the cat bootstrap is complete
        self.mad_hatter.execute_hook("after_cat_bootstrap")

    def load_db(self):
        # if there is no db, create it
        create_db_and_tables()

        # access db from instance
        self.db = get_db_session

    def load_natural_language(self):

        # LLM and embedder
        self.llm = self.mad_hatter.execute_hook("get_language_model")
        self.embedder = self.mad_hatter.execute_hook("get_language_embedder")

        # HyDE chain
        hypothesis_prompt = langchain.PromptTemplate(
            input_variables=["input"],
            template=self.mad_hatter.execute_hook("hypothetical_embedding_prompt"),
        )

        self.hypothetis_chain = langchain.chains.LLMChain(prompt=hypothesis_prompt, llm=self.llm)

        self.summarization_prompt = self.mad_hatter.execute_hook("summarization_prompt")

        # custom summarization chain
        self.summarization_chain = langchain.chains.LLMChain(
            llm=self.llm,
            verbose=False,
            prompt=langchain.PromptTemplate(template=self.summarization_prompt, input_variables=["text"]),
        )

        # set the default prompt settings
        self.default_prompt_settings = {
            "prefix": self.mad_hatter.execute_hook("agent_prompt_prefix"),
            "use_episodic_memory": True,
            "use_declarative_memory": True,
            "use_procedural_memory": True,
        }

    def load_memory(self):
        # Memory
        vector_memory_config = {"cat": self, "verbose": True}
        self.memory = LongTermMemory(vector_memory_config=vector_memory_config)
        self.working_memory = WorkingMemory()

    def load_plugins(self):
        # Load plugin system
        self.mad_hatter = MadHatter(self)

    def get_base_url(self):
        secure = os.getenv('CORE_USE_SECURE_PROTOCOLS', '')
        if secure != '':
            secure = 's'
        return f'http{secure}://{os.environ["CORE_HOST"]}:{os.environ["CORE_PORT"]}'

    def get_base_path(self):
        return os.path.join(os.getcwd(), "cat/")

    def get_plugin_path(self):
        return os.path.join(os.getcwd(), "cat/plugins/")

    def get_static_url(self):
        return self.get_base_url() + "/static"

    def get_static_path(self):
        return os.path.join(os.getcwd(), "cat/static/")

    def recall_relevant_memories_to_working_memory(self):

        user_message = self.working_memory["user_message_json"]["text"]
        prompt_settings = self.working_memory["user_message_json"]["prompt_settings"]

        # hook to do something before recall begins
        k, threshold = self.mad_hatter.execute_hook("before_cat_recalls_memories", user_message)

        # We may want to search in memory
        memory_query_text = self.mad_hatter.execute_hook("cat_recall_query", user_message)
        log(f'Recall query: "{memory_query_text}"')

        # embed recall query
        memory_query_embedding = self.embedder.embed_query(memory_query_text)
        self.working_memory["memory_query"] = memory_query_text

        if prompt_settings["use_episodic_memory"]:
            # recall relevant memories (episodic)
            episodic_memories = self.memory.vectors.episodic.recall_memories_from_embedding(
                embedding=memory_query_embedding, k=k, threshold=threshold
            )
        else:
            episodic_memories = []

        self.working_memory["episodic_memories"] = episodic_memories

        if prompt_settings["use_declarative_memory"]:
            # recall relevant memories (declarative)
            declarative_memories = self.memory.vectors.declarative.recall_memories_from_embedding(
                embedding=memory_query_embedding, k=k, threshold=threshold
            )
        else:
            declarative_memories = []

        self.working_memory["declarative_memories"] = declarative_memories

        # hook to modify/enrich retrieved memories
        self.mad_hatter.execute_hook("after_cat_recalled_memories", memory_query_text)

    def format_agent_executor_input(self):
        # format memories to be inserted in the prompt
        episodic_memory_formatted_content = self.mad_hatter.execute_hook(
            "agent_prompt_episodic_memories",
            self.working_memory["episodic_memories"],
        )
        declarative_memory_formatted_content = self.mad_hatter.execute_hook(
            "agent_prompt_declarative_memories",
            self.working_memory["declarative_memories"],
        )

        # format conversation history to be inserted in the prompt
        conversation_history_formatted_content = self.mad_hatter.execute_hook(
            "agent_prompt_chat_history", self.working_memory["history"]
        )

        return {
            "input": self.working_memory["user_message_json"]["text"],
            "episodic_memory": episodic_memory_formatted_content,
            "declarative_memory": declarative_memory_formatted_content,
            "chat_history": conversation_history_formatted_content,
            "ai_prefix": "AI",
        }

    def store_new_message_in_working_memory(self, user_message_json):

        # store last message in working memory
        self.working_memory["user_message_json"] = user_message_json

        prompt_settings = deepcopy(self.default_prompt_settings)

        # override current prompt_settings with prompt settings sent via websocket (if any)
        prompt_settings.update(user_message_json.get("prompt_settings", {}))

        self.working_memory["user_message_json"]["prompt_settings"] = prompt_settings

    def __call__(self, user_message_json):

        log(user_message_json, "INFO")

        # hook to modify/enrich user input
        user_message_json = self.mad_hatter.execute_hook("before_cat_reads_message", user_message_json)

        # store user_message_json in working memory
        # it contains the new message, prompt settings and other info plugins may find useful
        self.store_new_message_in_working_memory(user_message_json)

        # TODO another hook here?

        # recall episodic and declarative memories from vector collections
        #   and store them in working_memory
        try:
            self.recall_relevant_memories_to_working_memory()
        except Exception as e:
            log(e)
            traceback.print_exc(e)

            err_message = (
                "Vector memory error: you probably changed "
                "Embedder and old vector memory is not compatible. "
                "Please delete `core/long_term_memory` folder."
            )
            return {
                "error": False,
                # TODO: Otherwise the frontend gives notice of the error
                #   but does not show what the error is
                "content": err_message,
                "why": {},
            }

        # prepare input to be passed to the agent executor.
        #   Info will be extracted from working memory
        agent_executor_input = self.format_agent_executor_input()

        """
        # load agent (will rebuild both agent and agent_executor
        #   based on context and plugins)

        agent_executor = self.agent_manager.get_agent_executor()

        # reply with agent
        try:
            cat_message = agent_executor(agent_executor_input)
            print("Cat message total")
            print(cat_message)
        except Exception as e:
            # This error happens when the LLM
            #   does not respect prompt instructions.
            # We grab the LLM output here anyway, so small and
            #   non instruction-fine-tuned models can still be used.
            error_description = str(e)
            log("LLM does not respect prompt instructions", "ERROR")
            log(error_description, "ERROR")
            if not "Could not parse LLM output: `" in error_description:
                raise e

            unparsable_llm_output = error_description.replace("Could not parse LLM output: `", "").replace("`", "")
            cat_message = {
                "input": agent_executor_input["input"],
                "intermediate_steps": [],
                "output": unparsable_llm_output
            }

        log("cat_message:", "DEBUG")
        log(cat_message, "DEBUG")
"""

        """
        START @Colasuonno (Model change and using stuff chain)
        """

        """
        CHECK FOR QUESTION:
        
        If the question is like
        "Puoi ripetere?"
        "Non ho capito"
        "Approfondisci"
        
        I want to re-write this question in this way
        
        
        "Puoi ripetere?" -> "{old_question}"
        "Non ho capito" -> "Puoi spiegarmi dettagliatamente {old_question}"
        "Approfondisci" -> ...
        
        """

        prompt_question_format = """
                You are a question generator.
                You have to generate question with the same meaning starting from this input:

                Input: {question}

                If the question is too generic, you have to generate a question in this way: {question} + {last_question}

               Answer in Italian:
                """

        PROMPT_REFORMAT = PromptTemplate.from_template(prompt_question_format)
        chat_history = self.working_memory["history"]
        print("FORMATTED QUESTION")

        chain = LLMChain(llm=self.llm, prompt=PROMPT_REFORMAT)

        chain_info = {
            "question": agent_executor_input["input"],
            "last_question": chat_history[len(chat_history) - 2]["message"] if
                                           agent_executor_input[
                                               "chat_history"] != "" else
                                           agent_executor_input["input"]
        }
        print(chain_info)
       # formatted_question = chain.run(chain_info)

      #  print(formatted_question)

        #max_working_memory, max_question = self.analyze_every_questions_response(questions_formatted)

        # Changing the message input
       # self.working_memory["user_message_json"]["text"] = formatted_question


        """
        
        UNCOMMENT IF YOU WANT TO ENABLE THE FIRST CHAIN
        
         # Re-Defining declaring memory
        try:
            self.recall_relevant_memories_to_working_memory()
        except Exception as e:
            log(e)
            traceback.print_exc(e)

            err_message = (
                "Vector memory error: you probably changed "
                "Embedder and old vector memory is not compatible. "
                "Please delete `core/long_term_memory` folder."
            )
            return {
                "error": False,
                # TODO: Otherwise the frontend gives notice of the error
                #   but does not show what the error is
                "content": err_message,
                "why": {},
            }

        # Changing the message input
        #self.working_memory = max_working_memory
        #print("Max working declarative memory question is")
        #@print(max_question)


        print("Memory refactored")
        """



        # Reformat the agent input
        agent_executor_input = self.format_agent_executor_input()

        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,
         just say that you don't know, don't try to make up an answer. Your answer can't include questions.

                   {context}

                    
                   Question: {question}
                   Answer in Italian:"""

        print("Prompt template")
        print(prompt_template)

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=PROMPT)

        chain_info = {"input_documents": [
            Document(
                page_content="You have no information" if agent_executor_input["declarative_memory"] == "" else
                agent_executor_input["declarative_memory"],
                metadata={},
            )
        ], "question": agent_executor_input["input"],
        }

        cat_message = chain(chain_info)

        """
        END @Colasuonno
        """

        print("END CAT MESSAGE")
        print(cat_message)

        # update conversation history
        user_message = self.working_memory["user_message_json"]["text"]
        self.working_memory.update_conversation_history(who="Human", message=user_message)
        self.working_memory.update_conversation_history(who="AI", message=cat_message["output_text"])

        # store user message in episodic memory
        # TODO: vectorize and store also conversation chunks
        #   (not raw dialog, but summarization)
        _ = self.memory.vectors.episodic.add_texts(
            [user_message],
            [{"source": "user", "when": time.time()}],
        )

        # build data structure for output (response and why with memories)
        episodic_report = [dict(d[0]) | {"score": float(d[1])} for d in self.working_memory["episodic_memories"]]
        declarative_report = [dict(d[0]) | {"score": float(d[1])} for d in self.working_memory["declarative_memories"]]
        final_output = {
            "error": False,
            "type": "chat",
            "content": cat_message.get("output_text"),
            "why": {
                "input": cat_message.get("input"),
                "intermediate_steps": cat_message.get("intermediate_steps"),
                "memory": {
                    "episodic": episodic_report,
                    "declarative": declarative_report,
                },
            },
        }

        final_output = self.mad_hatter.execute_hook("before_cat_sends_message", final_output)

        return final_output



    def analyze_every_questions_response(self, questions):
            """
            This function is used to loop all the questions and get the most declarative
            Returns:
            """

            max_declarative = 0
            max_working_memory = None
            max_question = None

            for question in questions:

                if len(question) == 0 or type(question) is not str:
                    continue

                print("Analyzing " + str(question))

                # Changing the message input
                self.working_memory["user_message_json"]["text"] = question

                # Re-Defining declaring memory
                try:
                    self.recall_relevant_memories_to_working_memory()
                except Exception as e:
                    log(e)
                    traceback.print_exc(e)

                    err_message = (
                        "Vector memory error: you probably changed "
                        "Embedder and old vector memory is not compatible. "
                        "Please delete `core/long_term_memory` folder."
                    )
                    return {
                        "error": False,
                        # TODO: Otherwise the frontend gives notice of the error
                        #   but does not show what the error is
                        "content": err_message,
                        "why": {},
                    }

                declarative_len = len(self.working_memory["declarative_memories"])

                if max_declarative < declarative_len or not max_question:
                    max_declarative = declarative_len
                    max_working_memory = self.working_memory
                    max_question = question

            return max_working_memory, max_question + "?"

