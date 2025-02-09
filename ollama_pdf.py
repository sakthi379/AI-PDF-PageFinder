import PyPDF2
from openai import OpenAI
import os
fireworks_client = OpenAI(api_key=os.getenv('fireworks_api_key'), base_url='https://api.fireworks.ai/inference/v1')
ollama_client = OpenAI(api_key='ollama', base_url='http://192.168.29.2:11434/v1/')
lmstudio = OpenAI(api_key='ollama', base_url='http://127.0.0.1:1234/v1')
# gpt_client = OpenAI(api_key=openai_token)

list_index = []

def extract_bookmarks(pdf_path):
    index = {}
    count = 0
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        outlines = reader.outline  # Get the document outline

        def parse_outline(outlines, count, level=0):
            for item in outlines:
                if isinstance(item, list):
                    parse_outline(item, count, level + 1)  # Recursive call for nested bookmarks
                else:
                    title = item.title
                    page_number = reader.get_destination_page_number(item) + 1
                    # print("  " * level + f"{title} (Page {page_number})")
                    count += 1
                    list_index.append({"title": title, "page": page_number, "level": level})    
                    # index[count] = {"title": title, "page": page_number, "level": level}

        parse_outline(outlines, count)
        return index

def get_gpt4_response(client, model, prompt, index):
    try:
        # Call the OpenAI API
        messages = [
            {"role": "system", "content": "Based on user question, I will provide which possible page number to refer based on the question or the related topics of the question in the book of the bookmarks provided"},
            {"role": "system", "content": f"bookmark : {str(index)}"},
            {"role": "user", "content": f"only provide answers in json format list: ['start' and 'end'] based on likely end of the topic in the book, include multiple entries for related topics if necessary"},
            {"role": "user", "content": f"{prompt}"}
        ]
        # Return necessary index required for the response based on the prompt
        response = client.chat.completions.create(
            model=model,
            messages=messages, temperature=0.2, stream=True
        )
        # Extract the response text
        full_response = ""
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content or ""
            print(chunk_content, end="", flush=True)
            full_response += chunk_content
        return full_response
    except Exception as e:
        print(f"Error: {e}")
        quit()
        return None

# Example usage
pdf = R"C:\Users\sakthi\OneDrive\books\radiodiagnosis\AIIMS_MAMC_Pgi's_Comprehensive_Textbook_of_Diagnostic_Radiology.pdf"
index = extract_bookmarks(pdf)
print(len(list_index))
prompt = "I need information on 'neuroblastoma'"
response = get_gpt4_response(lmstudio, "meta-llama-3.1-8b-instruct", prompt, list_index)
# response = get_gpt4_response(gpt_client, "gpt-4o-mini", prompt, list_index)

# print(response)