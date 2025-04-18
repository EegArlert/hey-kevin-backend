from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv
import os, json, time
import sys
sys.path.append("..")
from middleware import api_key_middleware

load_dotenv()

API_KEY = str(os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=API_KEY)

app = FastAPI()
api_key_middleware(app)

# Explaining chatgpt how to spit out results
comments_generate_message = """"Give me a JSON output (do not include ANY other sentence. I strictly need JSON output. 
     do not include any whitespace or newline specifiers.
     Generate three different styles of quotes (sarcastic, witty, and funny) about an item I give you next.
     If you struggle to generate the output, you must say "Nobody is home").
     Format strictly as: {"sarcastic": "text", "witty": "text", "funny": "text"}.
     Try refraining from using "Oh great," and "Ah, yes" each time.
     If you get a prompt with no words and numbers no matter the size, don't generate a response and throw an error"""

# This function returns 3 comments generated from chatgpt. It runs in the background indefinitely, taking prompts after prompts,
# until you type 'exit'
def get_gpt_comments():
    client = OpenAI(api_key= API_KEY)

    # Setting up GPT
    set_up_gpt_message = comments_generate_message

    # Setting up input message
    messages = [{"role": "user", "content": set_up_gpt_message}]

    # While loop for continuous prompts
    while True:
        user_input = input("Give an item: (type 'exit' to stop): ")
        if user_input.lower() == "exit":
            break

        # Append user message to input message
        messages.append({"role": "user", "content":user_input})

        # To catch errors
        try:
            start_time = time.time() # for testing API response times

            # Get response
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=1.5,
                max_tokens=100
            )

            # JSON is located in completion.choices[0].message.content
            json_result = completion.choices[0].message.content

            # Print them out individually, Sarcastic, Witty, Funny, from the JSON that is provided by the API
            parsed_result = json.loads(json_result)
            for i in parsed_result:
                print("\nGenerated", i, "comment", ":", parsed_result[i])
            print("\n")

            #  Testing API response times
            response_time = time.time() - start_time
            print("API responded in", round(response_time, 2), "seconds\n")

        # If there is an exception, we display a fixed prompt
        except Exception as e:
            print(e)
            parsed_result = {"sarcastic": "Error 404: Sarcasm not found. Try again later.", "witty": "Critical failure: Wit module has crashed. Rebooting… never.", "funny": "System malfunction: Humor drive corrupted. Attempting emergency joke recovery… failed."}
            for i in parsed_result:
                print("\nGenerated", i, "comment", ":", parsed_result[i])
            print("\n")
            continue

        # Pop last prompt: Testing for API response delays
        messages.pop(1)

# Create server and map it via fastapi
@app.post("/comment")
async def comment(request: Request):

    set_up_gpt_message = comments_generate_message

    try:
        start_time = time.time() # for testing API response times
        
        data = await request.json()
        object_title = data.get("label")
        
        print(object_title)

        # Setting up input message
        messages = [{"role": "user", "content": set_up_gpt_message},
                    ({"role": "user", "content": object_title})]

        # Get response
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=1.5,
            max_tokens=100
        )

        # JSON is located in completion.choices[0].message.content
        json_result = completion.choices[0].message.content

        # Print them out individually, Sarcastic, Witty, Funny, from the JSON that is provided by the API
        parsed_result = json.loads(json_result)

        
        for i in parsed_result:
            print("\nGenerated", i, "comment", ":", parsed_result[i])
        print("\n")

        #  Testing API response times
        response_time = time.time() - start_time
        print("API responded in", round(response_time, 2), "seconds")
        
        return JSONResponse(content=parsed_result)

    # If there is an exception, we display a fixed prompt
    except Exception as e:
        print(e)
        parsed_result = {"sarcastic": "Error 404: Sarcasm not found. Try again later.", "witty": "Critical failure: Wit module has crashed. Rebooting… never.", "funny": "System malfunction: Humor drive corrupted. Attempting emergency joke recovery… failed."}
        for i in parsed_result:
            print("\nGenerated", i, "comment", ":", parsed_result[i])
        print("\n")
    return parsed_result
