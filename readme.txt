The display_msg function displays the message passed to it as the msg argument. The author argument determines who the author of the message is - it can be either ‘user’ or ‘assistant’.

Here’s how it works:

st.session_state.messages.append({"role": author, "content": msg}): This line appends the message to the session state. The message is stored as a dictionary with two keys: ‘role’ and ‘content’. ‘role’ is the author of the message, and ‘content’ is the message itself.
st.chat_message(author).write(msg): This line writes the message to the Streamlit chat interface. The author of the message is specified, and the message content is written to the chat.
So, the message that gets displayed is the one you pass to this function via the msg argument. The author argument determines whether the message is displayed as coming from the user or the assistant.
