import streamlit as st
import ollama

# Set the model
model = 'mistral:latest'

# Title of the app
st.title("Prompt Refiner with Local Ollama Model")

# Function to generate a response from the model
def generate_response(prompt):
    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

# Function to get prompt improvement suggestions
def get_prompt_improvement(original_prompt, response):
    improvement_prompt = f"Based on the following prompt and response, suggest how to improve the prompt to get a better response:\n\nPrompt: {original_prompt}\nResponse: {response}"
    improvement_response = ollama.chat(model=model, messages=[{'role': 'user', 'content': improvement_prompt}])
    return improvement_response['message']['content']

# Function to generate prompt variations (optional feature)
def generate_prompt_variations(base_prompt, num_variations=3):
    variation_prompt = f"Generate {num_variations} different ways to phrase this prompt: {base_prompt}"
    variations_response = ollama.chat(model=model, messages=[{'role': 'user', 'content': variation_prompt}])
    variations = variations_response['message']['content'].split('\n')  # Assuming variations are separated by new lines
    return [var.strip() for var in variations if var.strip()]  # Clean up and filter empty lines

# Streamlit form for user input
with st.form("my_form"):
    user_prompt = st.text_area("Enter your prompt:", "Describe your task here (e.g., 'Write a Python function to sort a list').")
    include_variations = st.checkbox("Generate prompt variations for comparison", value=False)
    submitted = st.form_submit_button("Submit")

# Process the form submission
if submitted:
    with st.spinner("Generating response..."):
        # Step 1: Get the model's response to the user's prompt
        response = generate_response(user_prompt)
        st.write("### Model's Response")
        st.write(response)
        
        # Step 2: Get suggestions for improving the prompt
        with st.spinner("Getting prompt improvement suggestions..."):
            improvement = get_prompt_improvement(user_prompt, response)
            st.write("### Suggestions to Improve Your Prompt")
            st.write(improvement)

        # Step 3: Optional - Generate and show prompt variations
        if include_variations:
            with st.spinner("Generating prompt variations..."):
                variations = generate_prompt_variations(user_prompt)
                st.write("### Prompt Variations and Responses")
                for i, var in enumerate(variations[:3], 1):  # Limit to 3 for performance
                    st.write(f"**Variation {i}:** {var}")
                    var_response = generate_response(var)
                    st.write(f"**Response for Variation {i}:**")
                    st.write(var_response)

# Sidebar for instructions
st.sidebar.header("How to Use")
st.sidebar.write("""
- Enter a prompt in the text area (e.g., "Write a Python function to calculate factorial").
- Click "Submit" to see the model's response and suggestions for improving your prompt.
- Check "Generate prompt variations" to see different phrasings and their responses, helpful for learning optimal prompt crafting.
- This app is particularly useful for refining prompts for coding tasks, enhancing interaction with the model.
""")