import discord
import os
import google.generativeai as genai
import faiss
import numpy as np
# `dotenv` is not strictly needed if using Replit Secrets, but the import is harmless.
# We will NOT call load_dotenv() as Replit injects secrets automatically.
#from dotenv import load_dotenv
import time  # Keep for potential future use (e.g., advanced rate limiting)

# --- Configuration Loading (using Replit Secrets via os.getenv) ---

# These MUST be set in the Replit "Secrets" tab (padlock icon)
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# Set a default model, but configuring in Secrets is better if you need a specific one
AI_MODEL_NAME = os.getenv('AI_MODEL_NAME')

# --- Essential Checks ---
if not DISCORD_TOKEN:
    print("FATAL ERROR: DISCORD_BOT_TOKEN not found.")
    print("Please set this in the Replit Secrets tab.")
    exit()
if not GOOGLE_API_KEY:
    print("FATAL ERROR: GOOGLE_API_KEY not found.")
    print("Please set this in the Replit Secrets tab.")
    exit()

# --- RAG Configuration & Loading ---
# These file names must match exactly what you uploaded to Replit
INDEX_PATH = "pdf_rules_index.faiss"
TEXT_CHUNKS_PATH = "pdf_text_chunks.txt"
# Ensure this matches the model used during indexing (in build_index.py)
EMBEDDING_MODEL = 'models/embedding-001'  # Or your chosen embedding model
NUM_RESULTS_TO_RETRIEVE = 7  # How many relevant chunks to fetch (tune this)

# --- Load FAISS index and Text Chunks ---
# This code expects the files to be in the root directory of the Repl.
try:
    print(f"Attempting to load FAISS index from: {INDEX_PATH}")
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index file not found at '{INDEX_PATH}'. Please upload it.")
    index = faiss.read_index(INDEX_PATH)
    print(f"FAISS index loaded successfully with {index.ntotal} vectors.")

    print(f"Attempting to load text chunks from: {TEXT_CHUNKS_PATH}")
    if not os.path.exists(TEXT_CHUNKS_PATH):
        raise FileNotFoundError(
            f"Text chunks file not found at '{TEXT_CHUNKS_PATH}'. Please upload it."
        )
    with open(TEXT_CHUNKS_PATH, 'r', encoding='utf-8') as f:
        # Assuming one chunk per line, matching the saving logic in build_index.py
        text_chunks_list = [line.strip() for line in f.readlines()]
    print(
        f"Text chunks loaded successfully. Found {len(text_chunks_list)} chunks."
    )

    # --- Verification Step ---
    if index.ntotal != len(text_chunks_list):
        print("\n!!! CRITICAL WARNING !!!")
        print(
            f"Mismatch detected: FAISS index has {index.ntotal} vectors, but loaded {len(text_chunks_list)} text chunks."
        )
        print(
            "This indicates a problem with your uploaded index/chunk files. The bot may retrieve incorrect context or crash."
        )
        print(
            "Please ensure both files were generated together and uploaded correctly."
        )
        print("!!! BOT MAY NOT FUNCTION CORRECTLY !!!\n")
        # Consider adding exit() here if this mismatch is unacceptable
        # exit()

except FileNotFoundError as fnf_error:
    print(f"\nFATAL ERROR: {fnf_error}")
    print(
        "Ensure both index and chunk files are uploaded to the root directory of your Replit project."
    )
    exit()
except Exception as e:
    print(f"\nFATAL ERROR: Failed to load FAISS index or text chunks: {e}")
    print("Check file integrity and permissions.")
    exit()

# --- Google AI Client Configuration ---
try:
    print("Configuring Google AI Client...")
    genai.configure(api_key=GOOGLE_API_KEY)

    # --- Generation Config (Adjust temperature as desired) ---
    generation_config = {
        "temperature":
        0.6,  # Lower temp for more factual, allows some creativity for examples
        "top_p": 1.0,
        "top_k": 1,
        "max_output_tokens": 5000,  # Max length of the generated response
    }
    print(f"Generation Config: {generation_config}")

    # --- Safety Settings (Disabled as requested - USE WITH EXTREME CAUTION) ---
    # WARNING: Disabling safety settings can lead to harmful, unethical, or offensive content.
    # You are responsible for the bot's output.
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]
    print(
        "!!! WARNING: AI safety settings are configured to BLOCK_NONE. Monitor bot output carefully. !!!"
    )

    # --- Initialize the Generative Model ---
    print(f"Initializing Google AI Model: {AI_MODEL_NAME}...")
    model = genai.GenerativeModel(model_name=AI_MODEL_NAME,
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    print("Google AI Model initialized successfully.")

except Exception as e:
    print(f"FATAL ERROR: Failed to initialize Google AI Model: {e}")
    exit()

# --- Discord Client Setup ---
print("Setting up Discord Client...")
# Enable necessary intents for reading messages
intents = discord.Intents.default()
intents.message_content = True  # REQUIRED to read message content
intents.messages = True
print(
    f"Discord Intents: message_content={intents.message_content}, messages={intents.messages}"
)

client = discord.Client(intents=intents)
print("Discord Client created.")

# --- RAG Helper Functions ---


def retrieve_relevant_chunks(query, k=NUM_RESULTS_TO_RETRIEVE):
    """Embeds query, searches FAISS index, returns top k text chunks."""
    if not query:
        print("Warning: Empty query received in retrieve_relevant_chunks.")
        return []
    try:
        # Embed the user's query - Use 'RETRIEVAL_QUERY' task type
        print(f"Embedding query (first 50 chars): '{query[:50]}...'")
        query_embedding_result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="RETRIEVAL_QUERY"  # Important for search queries
        )
        query_embedding = np.array([query_embedding_result['embedding']
                                    ]).astype('float32')
        print(
            f"Query embedded successfully (dimension: {query_embedding.shape[1]})."
        )

        # Search the FAISS index
        print(f"Searching FAISS index for {k} nearest neighbors...")
        distances, indices = index.search(query_embedding, k)
        print(f"FAISS search completed. Found indices: {indices[0]}")

        # Retrieve the corresponding text chunks using the indices
        retrieved_chunks = []
        # Filter out invalid indices robustly
        valid_indices = [
            i for i in indices[0] if 0 <= i < len(text_chunks_list)
        ]
        if len(valid_indices) < len(indices[0]):
            print(
                f"Warning: Some retrieved indices were out of bounds ({len(indices[0]) - len(valid_indices)} ignored). Original indices: {indices[0]}"
            )

        retrieved_chunks = [text_chunks_list[i] for i in valid_indices]

        print(f"Retrieved {len(retrieved_chunks)} valid chunks.")
        return retrieved_chunks

    except Exception as e:
        print(f"ERROR during RAG retrieval for query '{query[:50]}...': {e}")
        # Consider logging the full exception traceback here in production
        # import traceback
        # print(traceback.format_exc())
        return []  # Return empty list on error


# Inside bot.py / main.py


def create_augmented_prompt(query, context_chunks):
    """Combines retrieved context with the user query into a prompt for the LLM,
       including instructions for detail and length control."""

    # --- Define the target character limit for the prompt ---
    # Aim slightly below Discord's 2000 limit to provide a buffer.
    TARGET_CHAR_LIMIT = 1900

    if not context_chunks:
        # If no relevant chunks were found, instruct the LLM accordingly.
        print(
            "No relevant context chunks found for query. Creating 'cannot answer' prompt."
        )
        # Keep this prompt concise as well.
        prompt = f"""You are an assistant answering questions based *only* on a specific rules document.
        You were asked the following question:
        "{query}"

        However, after searching the document, no relevant excerpts could be found to answer this specific question.
        Please state clearly that you cannot answer the question based on the provided document excerpts.
        Keep your response concise and under {TARGET_CHAR_LIMIT} characters."""
        return prompt

    # Join the retrieved chunks into a single string with clear separation
    context_str = "\n\n---\n\n".join(context_chunks)

    # --- Construct the final prompt with DETAILED INSTRUCTIONS ---
    # Emphasize using ONLY the provided context AND the length constraint.
    prompt = f"""You are an expert assistant answering questions based *ONLY* on the document excerpts provided below.
    Your primary goal is to provide accurate, **detailed, and comprehensive** answers derived *solely* from the text given.
    Do not use any external knowledge, prior training data, or assumptions beyond these excerpts. Explain the relevant points thoroughly.

    If the excerpts contain the information needed to answer the question, provide a comprehensive answer based strictly on that information.
    If the excerpts do not contain the information needed, you MUST explicitly state that the answer cannot be found in the provided excerpts.

    **CRITICAL INSTRUCTION:** Format your response appropriately for a chat message. Provide as much detail as possible from the excerpts, but **ensure your final answer does NOT exceed {TARGET_CHAR_LIMIT} characters in total length.** Be thorough but mindful of this length limit.

    --- Start of Relevant Document Excerpts ---

    {context_str}

    --- End of Relevant Document Excerpts ---

    Based *ONLY* on the excerpts provided above, and keeping the total response length under {TARGET_CHAR_LIMIT} characters, please provide a detailed answer to the following question:

    Question: {query}

    Answer:"""

    print(
        f"Created augmented prompt (Context length: {len(context_str)} chars, Target response: <{TARGET_CHAR_LIMIT} chars)."
    )
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
    print('RAG Bot is online and ready.')
    print('------')
    # Set bot status
    try:
        await client.change_presence(activity=discord.Game(
            name="Reading the rules"))
        print("Bot presence set successfully.")
    except Exception as e:
        print(f"Warning: Could not set bot presence - {e}")


@client.event
async def on_message(message):
    """Handles incoming messages."""

    # 1. Ignore messages from the bot itself to prevent loops
    if message.author == client.user:
        return

    # 2. Define how the bot is triggered (mention or prefix)
    bot_mention = f'<@{client.user.id}>'
    command_prefix = "!rule "  # Example prefix - change or set to None
    user_query = None

    # Check for mention first
    if message.content.startswith(bot_mention):
        user_query = message.content[len(bot_mention):].strip()
        print(
            f"\nReceived mention from {message.author}: '{user_query[:100]}...'"
        )
    # Check for prefix (only if one is defined)
    elif command_prefix and message.content.startswith(command_prefix):
        user_query = message.content[len(command_prefix):].strip()
        print(
            f"\nReceived command from {message.author}: '{user_query[:100]}...'"
        )

    # 3. Process the query if the bot was triggered
    if user_query:
        # Indicate activity in Discord channel
        async with message.channel.typing():
            try:
                # ===== RAG Pipeline Execution =====
                start_time = time.time()  # Optional: timing

                # 1. Retrieve relevant context chunks
                retrieved_chunks = retrieve_relevant_chunks(user_query)

                # 2. Create the augmented prompt
                augmented_prompt = create_augmented_prompt(
                    user_query, retrieved_chunks)
                # DEBUG: Uncomment to see the full prompt sent to the LLM in Replit console
                # print(f"\n--- Augmented Prompt Sent [Length: {len(augmented_prompt)}] ---\n{augmented_prompt}\n-----------------------------------\n")

                # 3. Generate response using the Google AI model
                print("Generating content with Google AI...")
                response = await model.generate_content_async(augmented_prompt)

                # Extract the text response
                ai_response_text = response.text
                end_time = time.time()  # Optional: timing
                print(
                    f"AI response received (Length: {len(ai_response_text)} chars). Time taken: {end_time - start_time:.2f}s"
                )
                # print(f"Full AI Response Text:\n{ai_response_text}\n") # DEBUG: Uncomment to see full response

                # ===== Send Response to Discord =====
                if not ai_response_text or not ai_response_text.strip():
                    print(
                        "Warning: Received empty or whitespace-only response from AI."
                    )
                    await message.channel.send(
                        "I received an empty response. Please try rephrasing your question."
                    )
                # Handle Discord's 2000 character limit per message
                elif len(ai_response_text) > 1990:
                    print(
                        "Response too long for Discord, sending truncated version."
                    )
                    # Send the first part
                    await message.channel.send(ai_response_text[:1990] +
                                               "\n... (message truncated)")
                    # Optional: Send subsequent parts in new messages (more complex logic)
                else:
                    # Send the complete response
                    await message.channel.send(ai_response_text)
                print("Response sent to Discord.")

            # --- Robust Error Handling for the RAG/Generation Process ---
            except Exception as e:
                print(
                    f"\n!!! ERROR processing query for '{user_query[:50]}...' !!!"
                )
                # Log the full traceback for debugging in Replit console
                import traceback
                print(traceback.format_exc())

                # Check for specific feedback from the API if available
                safety_feedback_text = ""
                try:
                    # Access safety feedback if the response object exists and has it
                    if 'response' in locals(
                    ) and response and response.prompt_feedback:
                        safety_feedback_text = f" Safety Feedback: {response.prompt_feedback}"
                        print(safety_feedback_text)
                except AttributeError:
                    print(
                        "(Safety feedback attribute not found on response object)"
                    )
                except Exception as feedback_e:
                    print(f"(Error retrieving safety feedback: {feedback_e})")

                # Send a user-friendly error message back to Discord
                error_message = (
                    f"Sorry, I encountered an error while processing that request based on the document. "
                    f"Please try rephrasing or contact the administrator. "
                    f"(Error Type: {type(e).__name__})"
                    f"{safety_feedback_text}")  # Append safety feedback if any
                try:
                    await message.channel.send(error_message)
                except Exception as send_error:
                    print(
                        f"Failed to send error message to Discord: {send_error}"
                    )


# --- Run the Bot ---
print("\nAttempting to start the Discord bot...")
try:
    client.run(DISCORD_TOKEN)
except discord.errors.LoginFailure:
    print("\nFATAL ERROR: Invalid Discord Bot Token.")
    print("Please double-check the DISCORD_BOT_TOKEN in your Replit Secrets.")
except discord.errors.PrivilegedIntentsRequired:
    print(
        "\nFATAL ERROR: Required Privileged Intents (like Message Content) are not enabled."
    )
    print(
        "Go to your Discord Developer Portal -> Your Application -> Bot section."
    )
    print(
        "Scroll down to 'Privileged Gateway Intents' and ENABLE 'Message Content Intent'."
    )
except Exception as e:
    # Catch any other unexpected errors during startup
    print(
        f"\nFATAL ERROR: An unexpected error occurred while starting the bot: {e}"
    )
    import traceback
    print(traceback.format_exc())

print("\nBot process has ended.")
