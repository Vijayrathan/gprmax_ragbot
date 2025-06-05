from chatbot_components.chat_history import get_chat_history,store_chat_history
from chatbot_components.component_init import chat_client

QA_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in GPRMAX software.
You are speaking in a friendly and professional manner.
You should not answer questions that are not related to GPRMAX software.
Your primary goal is to help users with GPR and GPRMax related questions.
Focus on providing clear, accurate information about parameters, concepts, and general queries.
Do not generate simulation files in this mode."""

SIMULATION_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in GPRMAX software.
You are speaking in a friendly and professional manner.
Your only goal is to generate GPRMax simulation files.
When users request a simulation, ask follow-up questions if you don't have all the necessary parameters in the context.
Once you have all required information, generate a complete .in file.

IMPORTANT INSTRUCTIONS:
1. Include ALL parameter variations found in the context, especially for PML schemes (pml_cfs) which may have multiple different implementations.
2. If multiple definitions of the same parameter are found (like multiple pml_cfs lines), include ALL of them in the generated file.
3. Include reference information and comments exactly as they appear in the original file.
4. Do not abbreviate or truncate the file content - include everything from the original file.

Format the simulation file in a code block with the following structure:
```gprmax
# filename: simulation.in
# Your simulation data here
```
Always separate the code block from the rest of the answer with a blank line before and after."""

def generate_response(state,prompt,chat_history):
    query = prompt.invoke({"question": state["question"], "context": state["context"]})
    mode = state.get("mode", "qa")
    
    print(f"Generating response in mode: {mode}")
    
    # Get recent chat history
    history = get_chat_history(chat_history,state["session_id"])
    
    # Select system prompt based on mode
    system_prompt = QA_SYSTEM_PROMPT if mode == "qa" else SIMULATION_SYSTEM_PROMPT
    
    # Build messages with history
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add chat history
    for item in history:
        messages.append({"role": "user", "content": item["question"]})
        messages.append({"role": "assistant", "content": item["answer"]})
    
    # Debug context
    print(f"Mode: {mode}, Context sources:")
    for i, ctx in enumerate(state["context"]):
        if isinstance(ctx, str) and len(ctx) > 100:
            print(f"  Context {i+1}: {ctx[:100]}...")
        else:
            print(f"  Context {i+1}: {ctx}")
    
    # Add current context and question
    messages.append({"role": "user", "content": f'Context:\n{state["context"]}\n\nQuestion: {query}'})
    
    try:
        response = chat_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages
        )
        # Safely access the response content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                answer = response.choices[0].message.content
                # Store in chat history
                store_chat_history(chat_history,state["session_id"], state["question"], answer)
                return {"answer": answer}
        return {"answer": "Sorry, I couldn't generate a response. Please try again."}
    except Exception as e:
        return {"answer": f"Error generating response: {str(e)}"}
