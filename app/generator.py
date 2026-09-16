from openai import OpenAI

client = OpenAI()

def build_prompt(question, chunks):
    description = """
Answer the question using only the provided context.

If the answer cannot be found in the context, say that the answer
cannot be found in the provided document.

Context:
"""

    contexts = []

    for chunk in chunks:
        page_numbers = chunk["page_numbers"]
        chunk_text = chunk["text"]

        context = f"""
        [Pages {", ".join(map(str, page_numbers))}]
        {chunk_text}
        """
        contexts.append(context)

    context_text = "\n".join(contexts)

    question_text = f"""
Question:
{question}
"""

    return description + context_text + question_text


def generate_answer(question, chunks):

    prompt = build_prompt(question, chunks)

    response = client.responses.create(

        model="gpt-5-nano",

        input=prompt,

    )

    return response.output_text
