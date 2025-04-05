# bot.py - Discord Bot with RAG using Google Gemini and FAISS
# DEPLOYED ON AZURE VM | Secrets via Azure Key Vault | Code via Git

# ADD THESE LINES AT THE TOP:
from dotenv import load_dotenv
print("Loading environment variables from .env file if present...")
load_dotenv() # Loads variables from .env into os.environ
# --------------------------

import discord
import os
import google.generativeai as genai
import faiss
import numpy as np
import time
import asyncio # Added for caching lock
# Required Azure SDK libraries for Key Vault access
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

print("--- Bot Script Starting ---")

# --- Rulebook Configuration ---
# Maps command prefix to its resources and name
RULEBOOKS = {
    "!swn": {
        "index_path": "swn_rules_index.faiss",   # Specific index for SWN
        "chunks_path": "swn_text_chunks.txt", # Specific chunks for SWN
        "name": "Stars Without Number (SWN)"
    },
    "!swa": {
        "index_path": "swade_rules_index.faiss", # Specific index for SWADE
        "chunks_path": "swade_text_chunks.txt", # Specific chunks for SWADE
        "name": "Savage Worlds Adventure Edition (SWADE)"
    },
    # Add new rulebooks here by adding a new prefix and its file paths
    # "!xyz": {
    #     "index_path": "xyz_rules_index.faiss",
    #     "chunks_path": "xyz_text_chunks.txt",
    #     "name": "XYZ Rulebook"
    # }
}
print(f"Defined rulebooks: {list(RULEBOOKS.keys())}")
# --- End Rulebook Configuration ---

# --- Global Cache for Loaded Rulebook Data ---
# Stores {'prefix': {'index': faiss_index, 'chunks': list_of_chunks}}
loaded_rulebook_data = {}
cache_lock = asyncio.Lock() # To prevent race conditions when loading cache
# --- End Cache ---


# --- Configuration Loading from Azure Environment ---
# Key Vault URL is expected as an environment variable (set in systemd)
KEY_VAULT_URL = os.getenv('AZURE_KEYVAULT_URL')
if not KEY_VAULT_URL:
    print("FATAL ERROR: AZURE_KEYVAULT_URL environment variable not set.")
    print("Ensure this is defined in the systemd service file.")
    exit(1) # Exit with a non-zero code to indicate error
print(f"Using Key Vault URL: {KEY_VAULT_URL}")

# --- Authenticate to Azure and Fetch Secrets ---
# (Your existing Key Vault fetching code remains unchanged here)
try:
    print("Attempting to get Azure credentials...")
    credential = DefaultAzureCredential()
    print("Azure credential object obtained.")
    secret_client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)
    print("Azure Key Vault Secret Client initialized.")

    DISCORD_TOKEN = secret_client.get_secret("discord-bot-token").value
    print("Successfully fetched Discord Token from Key Vault.")
    GOOGLE_API_KEY = secret_client.get_secret("google-api-key").value
    print("Successfully fetched Google API Key from Key Vault.")

    try:
        AI_MODEL_NAME = secret_client.get_secret("ai-model-name").value
        print(f"Using AI Model Name from Key Vault: {AI_MODEL_NAME}")
    except Exception:
        print("AI Model Name secret not found in Key Vault, using default.")
        AI_MODEL_NAME = 'gemini-1.5-flash' # Adjusted default based on previous context
        print(f"Defaulting AI Model Name to: {AI_MODEL_NAME}")


    if not DISCORD_TOKEN or not GOOGLE_API_KEY:
        raise ValueError("Fetched secrets are empty or invalid.")

except Exception as e:
    print(f"FATAL ERROR: Failed to authenticate or retrieve secrets from Azure Key Vault: {e}")
    if "AuthorizationFailed" in str(e) or "Forbidden" in str(e) or "does not have secrets get permission" in str(e):
         print("Hint: Verify the VM's Managed Identity has the 'Key Vault Secrets User' role assigned on the target Key Vault.")
    elif "CredentialUnavailableError" in str(e):
         print("Hint: Ensure the VM has a System-Assigned Managed Identity enabled and it's being picked up.")
    exit(1)
# --- End Secrets Fetching ---


# --- RAG Configuration (Global Settings) ---
# Settings that apply across rulebooks (can be overridden if needed)
EMBEDDING_MODEL = 'models/embedding-001' # Or your chosen embedding model compatible with indices
NUM_RESULTS_TO_RETRIEVE = 10
TARGET_CHAR_LIMIT = 1900 # Target length for AI response
print(f"Global RAG Config: Embedding Model={EMBEDDING_MODEL}, k={NUM_RESULTS_TO_RETRIEVE}, Target Chars={TARGET_CHAR_LIMIT}")

# !!! NOTE: Global loading of INDEX_PATH/TEXT_CHUNKS_PATH is REMOVED !!!
# Loading now happens dynamically based on the command prefix


# --- Google AI Client Configuration ---
# (Your existing Google AI client setup remains unchanged here)
try:
    print("Configuring Google AI Client...")
    genai.configure(api_key=GOOGLE_API_KEY)

    # Generation config - Use a dictionary compatible with the SDK
    generation_config_dict = {
        "temperature": 0.8, # Slightly increased for potentially more varied answers
        "top_p": 0.95,
        "top_k": 64, # Allow more flexibility than top_k=1
        "max_output_tokens": 8192, # Model's max output tokens (Gemini 1.5 Flash)
    }
    print(f"Generation Config: {generation_config_dict}")

    # Convert dictionary to GenerationConfig object if needed by your genai version, or pass directly
    # generation_config_obj = genai.types.GenerationConfig(**generation_config_dict) # Example if object needed

    safety_settings = [ {"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"] ]
    print("!!! WARNING: AI safety settings are configured to BLOCK_NONE. Monitor bot output carefully. !!!")

    print(f"Initializing Google AI Model: {AI_MODEL_NAME}...")
    # Pass config dict directly if supported, or use the config object
    model = genai.GenerativeModel(
        model_name=AI_MODEL_NAME,
        generation_config=generation_config_dict, # Pass the dictionary
        safety_settings=safety_settings
    )
    print("Google AI Model initialized successfully.")

except Exception as e:
    print(f"FATAL ERROR: Failed to initialize Google AI Model: {e}")
    exit(1)


# --- Discord Client Setup ---
# (Your existing Discord client setup remains unchanged here)
print("Setting up Discord Client...")
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True # Ensure messages intent is enabled
client = discord.Client(intents=intents)
print("Discord Client created.")


# --- Helper Function to Load Rulebook Data (with Caching) ---
async def get_rulebook_data(prefix, config):
    """Loads index and chunks for a rulebook, using cache if available."""
    async with cache_lock: # Ensure only one load operation per prefix at a time
        if prefix in loaded_rulebook_data:
            print(f"Using cached data for prefix '{prefix}'")
            return loaded_rulebook_data[prefix] # Return cached data

        print(f"Cache miss for '{prefix}'. Loading data for rulebook: {config['name']}...")
        index_path = config['index_path']
        chunks_path = config['chunks_path']

        try:
            # Check if files exist before trying to load
            if not os.path.exists(index_path):
                 raise FileNotFoundError(f"FAISS index file not found: '{os.getcwd()}/{index_path}'")
            if not os.path.exists(chunks_path):
                 raise FileNotFoundError(f"Text chunks file not found: '{os.getcwd()}/{chunks_path}'")

            print(f"Loading FAISS index from: ./{index_path}")
            index = faiss.read_index(index_path)
            print(f"FAISS index loaded ({index.ntotal} vectors).")

            print(f"Loading text chunks from: ./{chunks_path}")
            with open(chunks_path, 'r', encoding='utf-8') as f:
                text_chunks_list = [line.strip() for line in f.readlines()]
            print(f"Text chunks loaded ({len(text_chunks_list)} chunks).")

            # Verification Step
            if index.ntotal != len(text_chunks_list):
                print("\n!!! CRITICAL WARNING !!! Mismatch: FAISS index vectors vs loaded text chunks.")
                print(f"Rulebook: {config['name']}, Index: {index.ntotal}, Chunks: {len(text_chunks_list)}")
                print("Context retrieval may be incorrect. Verify index/chunk files are paired correctly.")
                # Decide if you want to cache mismatched data or raise an error preventing caching
                # For now, we'll cache it but log the warning. Consider returning None or raising error.

            # Store in cache
            loaded_rulebook_data[prefix] = {'index': index, 'chunks': text_chunks_list}
            print(f"Data for '{prefix}' loaded and cached.")
            return loaded_rulebook_data[prefix]

        except FileNotFoundError as fnf_error:
            print(f"\nERROR loading data for '{prefix}': {fnf_error}")
            # Don't cache failures
            return None # Indicate failure
        except Exception as e:
            print(f"\nERROR loading data for '{prefix}': Failed to load FAISS index or text chunks: {e}")
            # Don't cache failures
            return None # Indicate failure

# --- RAG Helper Functions (Modified) ---
def retrieve_relevant_chunks(query, index, text_chunks_list, k=NUM_RESULTS_TO_RETRIEVE):
    """Embeds query, searches the *provided* FAISS index, returns top k text chunks."""
    # Takes index and text_chunks_list as arguments now
    if not query or index is None or not text_chunks_list:
        print("Skipping retrieval due to missing query, index, or chunks.")
        return []
    try:
        print(f"Embedding query (first 50 chars): '{query[:50]}...' using {EMBEDDING_MODEL}")
        # Using the global EMBEDDING_MODEL here, ensure it's compatible with all indices
        query_embedding_result = genai.embed_content( model=EMBEDDING_MODEL, content=query, task_type="RETRIEVAL_QUERY" )
        query_embedding = np.array([query_embedding_result['embedding']]).astype('float32')

        print(f"Searching provided FAISS index ({index.ntotal} vectors) for {k} nearest neighbors...")
        distances, indices = index.search(query_embedding, k)

        # Filter out invalid indices (-1 or out of bounds)
        valid_indices = [i for i in indices[0] if 0 <= i < len(text_chunks_list)]
        retrieved_chunks = [text_chunks_list[i] for i in valid_indices]

        print(f"Retrieved {len(retrieved_chunks)} valid chunks.")
        return retrieved_chunks
    except Exception as e:
        print(f"ERROR during RAG retrieval for query '{query[:50]}...': {e}")
        # Consider logging the traceback here if errors persist
        # import traceback
        # print(traceback.format_exc())
        return []

def create_augmented_prompt(query, context_chunks, rulebook_name):
    """Combines retrieved context with the user query into a prompt, mentioning the rulebook."""
    # Takes rulebook_name as an argument now
    if not context_chunks:
        print(f"No relevant context chunks found for query related to {rulebook_name}. Creating 'cannot answer' prompt.")
        prompt = f"""You are an assistant answering questions based *only* on the {rulebook_name} rules document. You were asked: "{query}"
No relevant excerpts were found in the {rulebook_name} document. State clearly that you cannot answer based on the provided document excerpts. Keep your response concise and under {TARGET_CHAR_LIMIT} characters."""
        return prompt

    context_str = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are an expert assistant answering questions based *ONLY* on the {rulebook_name} document excerpts provided below. Provide accurate, **detailed, and comprehensive** answers derived *solely* from the text given. Do not use external knowledge. Explain relevant points thoroughly.
If excerpts contain the needed information, provide a comprehensive answer synthesizing information from all snippets if necessary. If not, state *explicitly* that the answer cannot be found in the provided {rulebook_name} excerpts.
**CRITICAL INSTRUCTION:** Format for a chat message. Be detailed, but **ensure your final answer does NOT exceed {TARGET_CHAR_LIMIT} characters total.**

--- Start of Relevant {rulebook_name} Document Excerpts ---
{context_str}
--- End of Relevant {rulebook_name} Document Excerpts ---

Based *ONLY* on the {rulebook_name} excerpts provided, and keeping the total response length under {TARGET_CHAR_LIMIT} characters, provide a detailed answer to:
Question: {query}
Answer:"""
    print(f"Created augmented prompt for {rulebook_name} (Context length: {len(context_str)} chars, Target response: <{TARGET_CHAR_LIMIT} chars).")
    return prompt

# --- Discord Event Handlers (Modified) ---
@client.event
async def on_ready():
    """Runs when the bot successfully connects to Discord."""
    print('------')
    print(f'Logged in as: {client.user.name} (ID: {client.user.id})')
    print(f'Discord.py Version: {discord.__version__}')
    print(f'Google AI Model: {AI_MODEL_NAME}')
    # Removed FAISS index count here as it's loaded dynamically
    # print(f'FAISS Index Vectors: {index.ntotal}') # REMOVED
    print(f'Working Directory: {os.getcwd()}')
    print('Multi-Rulebook RAG Bot (Azure VM / Key Vault) is online and ready.')
    print('------')
    try:
        # Update presence to reflect new commands
        rulebook_prefixes = " or ".join(RULEBOOKS.keys())
        await client.change_presence(activity=discord.Game(name=f"{rulebook_prefixes} + query"))
        print(f"Bot presence updated to: {rulebook_prefixes} + query")
    except Exception as e:
        print(f"Warning: Could not set bot presence - {e}")

@client.event
async def on_message(message):
    """Handles incoming messages based on mention or command prefix."""
    if message.author == client.user: return # Ignore self

    content = message.content.strip()
    bot_mention_formats = [f'<@{client.user.id}>', f'<@!{client.user.id}>'] # Handles nickname mention too

    # 1. Check for Direct Mention at the start
    mentioned = False
    for mention_format in bot_mention_formats:
        if content.startswith(mention_format):
             mentioned = True
             break

    if mentioned:
        print(f"Bot mentioned by {message.author.name}")
        help_message = f"Hi there! Please start your query with one of the following commands followed by your question:\n"
        for prefix, config in RULEBOOKS.items():
            help_message += f"- `{prefix}` for {config['name']}\n"
        help_message += f"Example: `{list(RULEBOOKS.keys())[0]} how does combat work?`"
        await message.channel.send(help_message)
        return # Stop processing after sending help

    # 2. Check for Rulebook Command Prefixes
    processed_command = False
    for prefix, config in RULEBOOKS.items():
        # Check for prefix + space (case-insensitive)
        if content.lower().startswith(prefix.lower() + " "):
            user_query = content[len(prefix):].strip()

            if not user_query:
                print(f"Received command prefix '{prefix}' with no query from {message.author.name}.")
                await message.channel.send(f"Please provide a question after the `{prefix}` command.")
                processed_command = True # Handled this command (it was empty)
                break # Stop checking other prefixes

            print(f"\nReceived command '{prefix}' for rulebook '{config['name']}' from {message.author.name}")
            print(f"Query: '{user_query[:100]}...'")
            processed_command = True # Mark as processed

            async with message.channel.typing():
                start_time = time.time()
                try:
                    # --- Get Rulebook Data (Load or from Cache) ---
                    rulebook_data = await get_rulebook_data(prefix, config)
                    if rulebook_data is None:
                        # Error loading data was already printed in get_rulebook_data
                        await message.channel.send(f"Sorry, I couldn't load the necessary data files for the {config['name']} rulebook. Please check the logs.")
                        break # Stop processing this command

                    current_index = rulebook_data['index']
                    current_chunks = rulebook_data['chunks']

                    # --- RAG Pipeline using specific rulebook data ---
                    retrieved_chunks = retrieve_relevant_chunks(user_query, current_index, current_chunks) # Pass loaded data
                    augmented_prompt = create_augmented_prompt(user_query, retrieved_chunks, config['name']) # Pass rulebook name

                    # --- Generate Content with Google AI ---
                    print("Generating content with Google AI...")
                    response = await model.generate_content_async(augmented_prompt) # Use the global model
                    ai_response_text = ""

                    # --- Process Response ---
                    # Check for safety blocks or empty response before accessing .text
                    try:
                        # Accessing parts or text might raise if the response was blocked.
                        ai_response_text = response.text
                    except ValueError as e:
                        # Handle cases where the response was blocked (often raises ValueError)
                        print(f"!!! WARNING: Gemini response generation likely blocked: {e}")
                        if response and response.prompt_feedback:
                             print(f"Safety Feedback: {response.prompt_feedback}")
                             safety_reason = str(response.prompt_feedback) # Get reason if available
                             await message.channel.send(f"I couldn't generate a response due to safety filters: {safety_reason}")
                        else:
                             await message.channel.send("I couldn't generate a response due to safety filters or an internal issue.")
                        break # Exit command processing
                    except Exception as e:
                         # Catch other potential errors accessing response parts
                         print(f"!!! ERROR accessing Gemini response content: {e}")
                         await message.channel.send("Sorry, there was an issue processing the AI's response.")
                         break # Exit command processing

                    end_time = time.time()
                    print(f"AI response received for {config['name']} (Length: {len(ai_response_text)} chars). Time: {end_time - start_time:.2f}s")

                    # --- Send Response (with truncation safety net) ---
                    if not ai_response_text or not ai_response_text.strip():
                        print("WARNING: Received empty or whitespace-only response from AI.")
                        await message.channel.send("I received an empty response from the AI. Please try rephrasing or check the context.")
                    elif len(ai_response_text) > 2000:
                        print("Response exceeds 2000 chars (AI likely ignored prompt limit), truncating.")
                        truncation_notice = "\n... (message truncated due to length limit)"
                        max_text_length = 2000 - len(truncation_notice)
                        await message.channel.send(ai_response_text[:max_text_length] + truncation_notice)
                    else:
                        await message.channel.send(ai_response_text)
                    print("Response sent to Discord.")

                except Exception as e:
                    # General error handling during the processing pipeline for a command
                    print(f"\n!!! ERROR processing command '{prefix}' for query '{user_query[:50]}...' !!!")
                    import traceback
                    print(traceback.format_exc()) # Log full traceback
                    await message.channel.send(f"Sorry, an unexpected error occurred while processing your request for {config['name']}. (Error: {type(e).__name__})")

            break # Important: Exit loop after processing the first matching command prefix

    # 3. Ignore messages that weren't mentions or processed commands
    if not mentioned and not processed_command:
        # print(f"Ignoring non-command/non-mention message: {content[:60]}...") # Optional debug
        pass

# --- Main Execution Block ---
# (Your existing main execution block remains unchanged here)
if __name__ == "__main__":
    print("\nAttempting to start the Discord client...")
    try:
        client.run(DISCORD_TOKEN) # Uses token fetched from Key Vault
    except discord.errors.LoginFailure:
        print("\nFATAL ERROR: Invalid Discord Bot Token.")
        print("Check the secret value in Azure Key Vault and the VM's permissions.")
        exit(1)
    except discord.errors.PrivilegedIntentsRequired:
        print("\nFATAL ERROR: Required Privileged Intents (Message Content/Messages) not enabled.")
        print("Enable in Discord Developer Portal -> Bot -> Privileged Gateway Intents.")
        exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected error occurred during bot execution: {e}")
        import traceback
        print(traceback.format_exc())
        exit(1)

    print("\nBot process has gracefully ended.")