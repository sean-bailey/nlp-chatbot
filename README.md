# nlp-chatbot
This is the repository to cover what has expanded from a thought experiment to something actually functional -- a chatbot using natural language processing with goals of ML integration with context.

This uses python3. To run it, (ensure you've got the appropriate libraries
installed),

`python3 wikichatbot.py`

It will prompt you for a topic to discuss. It's not quite intelligent -- if
you enter in something too complex, it will bork out. It basically just
appends the topic to the url of wikipedia articles.

It's fairly flexible, if you want to discuss a new topic, just type that in.

`Let's talk about something else!`,`I'm bored, give me a new topic.`,
`I want to discuss something different.`, etc.

It's got some _basic_ off-topic capabilities, can do basic acknowledgement of thank-you's, greetings, and to close the program down, simply tell it goodbye!
