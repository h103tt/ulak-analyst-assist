# ULAK Agent Assist


.lovable, public, src, supabase folders are mostly related with frontend. 

The main files about backend, model, rag etc. lie in the test_analysis_agent folder.

agent.py contains code for system prompt, model description and message listening.

vector_embed.py contains code for langchain integration, document uploading, chunking and embedding, also tools model to use when generating a response.

bridge.py mainly merges the backend and frontend

when you wanna run the model do the following

in ulak-analyst-assist folder run "bun run dev"

and in another terminal window after downloading requirements.txt
in the test-agent-env run "python bridge.py"

now you can access the ui chatbot in the http://localhost:8080/

