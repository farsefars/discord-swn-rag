# bot.py
# ADD THESE LINES AT THE TOP:
from dotenv import load_dotenv
print("Loading environment variables from .env file if present...")
load_dotenv() # Loads variables from .env into os.environ
# --------------------------

# bot.py - Discord Bot with RAG using Google Gemini and FAISS
# DEPLOYED ON AZURE VM | Secrets via Azure Key Vault | Code via Git

import discord
import os
import google.generativeai as genai
import faiss
import numpy as np
import time
# Required Azure SDK libraries for Key Vault access
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

print("--- Bot Script Starting ---")

# --- Configuration Loading from Azure Environment ---

# Key Vault URL is expected as an environment variable (set in systemd)
KEY_VAULT_URL = os.getenv('AZURE_KEYVAULT_URL')
if not KEY_VAULT_URL:
    print("FATAL ERROR: AZURE_KEYVAULT_URL environment variable not set.")
    print("Ensure this is defined in the systemd service file.")
    exit(1) # Exit with a non-zero code to indicate error
print(f"Using Key Vault URL: {KEY_VAULT_URL}")

# --- Authenticate to Azure and Fetch Secrets ---
try:
    # DefaultAzureCredential will automatically use the VM's Managed Identity when running in Azure
    # Ensure the VM's Managed Identity has 'Key Vault Secrets User' role on the Key Vault
    print("Attempting to get Azure credentials...")
    credential = DefaultAzureCredential()
    print("Azure credential object obtained.")
    secret_client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)
    print("Azure Key Vault Secret Client initialized.")

    # Fetch secrets using the names defined in Azure Key Vault
    DISCORD_TOKEN = secret_client.get_secret("discord-bot-token").value
    print("Successfully fetched Discord Token from Key Vault.")
    GOOGLE_API_KEY = secret_client.get_secret("google-api-key").value
    print("Successfully fetched Google API Key from Key Vault.")

    # Optional: Fetch model name if stored in Key Vault, otherwise use default
    try:
        AI_MODEL_NAME = secret_client.get_secret("ai-model-name").value
        print(f"Using AI Model Name from Key Vault: {AI_MODEL_NAME}")
    except Exception: # Catch specific exceptions like ResourceNotFoundError if needed
        print("AI Model Name secret not found in Key Vault, using default.")
        AI_MODEL_NAME = 'gemini-2.0-flash' # Default model

    # Basic validation after fetching
    if not DISCORD_TOKEN or not GOOGLE_API_KEY:
        raise ValueError("Fetched secrets are empty or invalid.")

except Exception as e:
    print(f"FATAL ERROR: Failed to authenticate or retrieve secrets from Azure Key Vault: {e}")
    # Add hints for common Managed Identity permission issues
    if "AuthorizationFailed" in str(e) or "Forbidden" in str(e) or "does not have secrets get permission" in str(e):
         print("Hint: Verify the VM's Managed Identity has the 'Key Vault Secrets User' role assigned on the target Key Vault.")
    elif "CredentialUnavailableError" in str(e):
         print("Hint: Ensure the VM has a System-Assigned Managed Identity enabled and it's being picked up.")
    # Include traceback for detailed debugging if needed (useful in initial setup)
    # import traceback
    # print(traceback.format_exc())
    exit(1) # Exit with a non-zero code
# --- End Secrets Fetching ---


# --- RAG Configuration & Loading ---
# Assumes index/chunks files are located in the same directory the script is run from
# (controlled by WorkingDirectory in systemd)
INDEX_PATH = "pdf_rules_index.faiss"
TEXT_CHUNKS_PATH = "pdf_text_chunks.txt"
EMBEDDING_MODEL = 'models/embedding-001' # Or your chosen embedding model
NUM_RESULTS_TO_RETRIEVE = 3

try:
    print(f"Loading FAISS index from: ./{INDEX_PATH}") # Use relative path notation
    if not os.path.exists(INDEX_PATH):
         raise FileNotFoundError(f"FAISS index file not found in working directory: '{os.getcwd()}/{INDEX_PATH}'. Ensure it exists.")
    index = faiss.read_index(INDEX_PATH)
    print(f"FAISS index loaded successfully ({index.ntotal} vectors).")

    print(f"Loading text chunks from: ./{TEXT_CHUNKS_PATH}") # Use relative path notation
    if not os.path.exists(TEXT_CHUNKS_PATH):
         raise FileNotFoundError(f"Text chunks file not found in working directory: '{os.getcwd()}/{TEXT_CHUNKS_PATH}'. Ensure it exists.")
    with open(TEXT_CHUNKS_PATH, 'r', encoding='utf-8') as f:
        text_chunks_list = [line.strip() for line in f.readlines()]
    print(f"Text chunks loaded successfully ({len(text_chunks_list)} chunks).")

    # Verification Step
    if index.ntotal != len(text_chunks_list):
        print("\n!!! CRITICAL WARNING !!! Mismatch: FAISS index vectors vs loaded text chunks.")
        print(f"Index vectors: {index.ntotal}, Chunks loaded: {len(text_chunks_list)}")
        print("Context retrieval may be incorrect. Verify index/chunk files are paired correctly.")
        # exit(1) # Consider exiting in production if this mismatch is critical

except FileNotFoundError as fnf_error:
    print(f"\nFATAL ERROR: {fnf_error}")
    print("Ensure index/chunk files are present in the bot's working directory.")
    exit(1)
except Exception as e:
    print(f"\nFATAL ERROR: Failed to load FAISS index or text chunks: {e}")
    exit(1)


# --- Google AI Client Configuration ---
try:
    print("Configuring Google AI Client...")
    genai.configure(api_key=GOOGLE_API_KEY) # Uses key fetched from KV

    generation_config = { "temperature": 0.6, "top_p": 1.0, "top_k": 1, "max_output_tokens": 5000, }
    print(f"Generation Config: {generation_config}")

    safety_settings = [ {"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"] ]
    print("!!! WARNING: AI safety settings are configured to BLOCK_NONE. Monitor bot output carefully. !!!")

    print(f"Initializing Google AI Model: {AI_MODEL_NAME}...")
    model = genai.GenerativeModel( model_name=AI_MODEL_NAME, generation_config=generation_config, safety_settings=safety_settings )
    print("Google AI Model initialized successfully.")

except Exception as e:
    print(f"FATAL ERROR: Failed to initialize Google AI Model: {e}")
    exit(1)


# --- Discord Client Setup ---
print("Setting up Discord Client...")
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
client = discord.Client(intents=intents)
print("Discord Client created.")


# --- RAG Helper Functions (retrieve_relevant_chunks, create_augmented_prompt) ---
# These functions remain unchanged as their logic is independent of the hosting platform
def retrieve_relevant_chunks(query, k=NUM_RESULTS_TO_RETRIEVE):
    """Embeds query, searches FAISS index, returns top k text chunks."""
    if not query: return []
    try:
        print(f"Embedding query (first 50 chars): '{query[:50]}...'")
        query_embedding_result = genai.embed_content( model=EMBEDDING_MODEL, content=query, task_type="RETRIEVAL_QUERY" )
        query_embedding = np.array([query_embedding_result['embedding']]).astype('float32')
        print(f"Searching FAISS index for {k} nearest neighbors...")
        distances, indices = index.search(query_embedding, k)
        valid_indices = [i for i in indices[0] if 0 <= i < len(text_chunks_list)]
        retrieved_chunks = [text_chunks_list[i] for i in valid_indices]
        print(f"Retrieved {len(retrieved_chunks)} valid chunks.")
        return retrieved_chunks
    except Exception as e:
        print(f"ERROR during RAG retrieval for query '{query[:50]}...': {e}")
        return []

def create_augmented_prompt(query, context_chunks):
    """Combines retrieved context with the user query into a prompt for the LLM, including length constraints."""
    TARGET_CHAR_LIMIT = 1900 # Target length for AI response
    if not context_chunks:
        print("No relevant context chunks found for query. Creating 'cannot answer' prompt.")
        prompt = f"""You are an assistant answering questions based *only* on a specific rules document. You were asked: "{query}"
        No relevant excerpts were found. State clearly that you cannot answer based on the provided document excerpts. Keep your response concise and under {TARGET_CHAR_LIMIT} characters."""
        return prompt

    context_str = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are an expert assistant answering questions based *ONLY* on the document excerpts provided below. Provide accurate, **detailed, and comprehensive** answers derived *solely* from the text given. Do not use external knowledge. Explain relevant points thoroughly.
    If excerpts contain the needed information, provide a comprehensive answer. If not, state *explicitly* that the answer cannot be found in the provided excerpts.
    **CRITICAL INSTRUCTION:** Format for a chat message. Be detailed, but **ensure your final answer does NOT exceed {TARGET_CHAR_LIMIT} characters total.**

    --- Start of Relevant Document Excerpts ---
    {context_str}
    --- End of Relevant Document Excerpts ---

    Based *ONLY* on the excerpts provided, and keeping the total response length under {TARGET_CHAR_LIMIT} characters, provide a detailed answer to:
    Question: {query}
    Answer:"""
    print(f"Created augmented prompt (Context length: {len(context_str)} chars, Target response: <{TARGET_CHAR_LIMIT} chars).")
    return prompt

# --- Discord Event Handlers ---
@client.event
async def on_ready():
    """Runs when the bot successfully connects to Discord."""
    print('------')
    print(f'Logged in as: {client.user.name} (ID: {client.user.id})')
    print(f'Discord.py Version: {discord.__version__}')
    print(f'Google AI Model: {AI_MODEL_NAME}')
    print(f'FAISS Index Vectors: {index.ntotal}')
    print(f'Working Directory: {os.getcwd()}') # Confirm working directory
    print('RAG Bot (Azure VM / Key Vault) is online and ready.')
    print('------')
    try:
        await client.change_presence(activity=discord.Game(name="Consulting Kevin Crawford"))
        print("Bot presence updated.")
    except Exception as e:
        print(f"Warning: Could not set bot presence - {e}")

@client.event
async def on_message(message):
    """Handles incoming messages based on mention or prefix."""
    if message.author == client.user: return # Ignore self

    bot_mention = f'<@{client.user.id}>'
    command_prefix = "!rule " # Adjust prefix if needed
    user_query = None

    if message.content.startswith(bot_mention):
        user_query = message.content[len(bot_mention):].strip()
        print(f"\nReceived mention from {message.author.name}: '{user_query[:100]}...'")
    elif command_prefix and message.content.startswith(command_prefix):
        user_query = message.content[len(command_prefix):].strip()
        print(f"\nReceived command from {message.author.name}: '{user_query[:100]}...'")

    if user_query:
        async with message.channel.typing():
            start_time = time.time()
            try:
                # RAG Pipeline
                retrieved_chunks = retrieve_relevant_chunks(user_query)
                augmented_prompt = create_augmented_prompt(user_query, retrieved_chunks)
                print("Generating content with Google AI...")
                response = await model.generate_content_async(augmented_prompt)
                ai_response_text = response.text
                end_time = time.time()
                print(f"AI response received (Length: {len(ai_response_text)} chars). Time: {end_time - start_time:.2f}s")

                # Send Response (with truncation safety net)
                if not ai_response_text or not ai_response_text.strip():
                    await message.channel.send("I received an empty response. Please try rephrasing.")
                elif len(ai_response_text) > 2000:
                    print("Response exceeds 2000 chars (AI likely ignored prompt limit), truncating.")
                    truncation_notice = "\n... (message truncated due to length limit)"
                    max_text_length = 2000 - len(truncation_notice)
                    await message.channel.send(ai_response_text[:max_text_length] + truncation_notice)
                else:
                    await message.channel.send(ai_response_text)
                print("Response sent to Discord.")

            except Exception as e:
                print(f"\n!!! ERROR processing query for '{user_query[:50]}...' !!!")
                import traceback
                print(traceback.format_exc()) # Log full traceback to journal
                safety_feedback_text = ""
                try: # Attempt to get safety feedback if response object exists
                    if 'response' in locals() and response and response.prompt_feedback:
                         safety_feedback_text = f" Safety Feedback: {response.prompt_feedback}"
                except Exception: pass # Ignore errors getting feedback
                await message.channel.send(f"Sorry, an error occurred. (Error: {type(e).__name__}){safety_feedback_text}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\nAttempting to start the Discord client...")
    try:
        client.run(DISCORD_TOKEN) # Uses token fetched from Key Vault
    except discord.errors.LoginFailure:
        print("\nFATAL ERROR: Invalid Discord Bot Token.")
        print("Check the secret value in Azure Key Vault and the VM's permissions.")
    except discord.errors.PrivilegedIntentsRequired:
        print("\nFATAL ERROR: Required Privileged Intents (Message Content) not enabled.")
        print("Enable in Discord Developer Portal -> Bot -> Privileged Gateway Intents.")
    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected error occurred during bot execution: {e}")
        import traceback
        print(traceback.format_exc())
        exit(1) # Ensure non-zero exit on critical error

    print("\nBot process has gracefully ended.")