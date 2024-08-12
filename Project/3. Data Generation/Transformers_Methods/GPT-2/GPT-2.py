# %%
!pip install -q transformers

# %%
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Install GPT-2 model and tokeniser
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# %%
import torch

# %%
def generate_answer(context, question):
    # Tokenise inputs
    inputs = tokenizer.encode(context + " " + question, return_tensors="pt")

    # Generate answer from model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove redundant text in the answer
    answer = answer.replace(context, "").replace(question, "").strip()

    return answer

def main():
    context = "Nuclear reactors are used for generating electricity through nuclear fission reactions."
    question = "What is the process of generating electricity in nuclear reactors?"
    
    # Produce the answer
    answer = generate_answer(context, question)
    
    # Create answer
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

# %%
def generate_answer(context, question):
    # Tokenize the inputs
    inputs = tokenizer.encode(context + " " + question, return_tensors="pt")

    # Generate the answer using the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the excess text from the answer
    answer = answer.replace(context, "").replace(question, "").strip()

    # Check for uniqueness at the sentence level
    sentences = [sent.capitalize().strip() for sent in answer.split('.') if sent.strip()]
    unique_words = set()
    unique_answer = []

    for sentence in sentences:
        words = sentence.split()
        unique_sentence = []
        for word in words:
            if word.lower() not in unique_words:
                unique_words.add(word.lower())
                unique_sentence.append(word)
        unique_answer.append(" ".join(unique_sentence))

    # Recombine the unique sentences
    processed_answer = ". ".join(unique_answer).strip() + "."

    return processed_answer

def main():
    context = "Nuclear reactors are used for generating electricity through nuclear fission reactions."
    question = "What is the process of generating electricity in nuclear reactors?"

    # Generate the answer
    answer = generate_answer(context, question)

    # Print the answer
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()


# %%


# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_question_and_answer(context):
    # Model and tokenize
    model_name = "gpt2-large"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)



# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_question_and_answer(context):
    # Parameters
    temperature = 1.0
    top_k = 50
    top_p = 0.95

    # Tokenize the context
    input_ids = tokenizer.encode(context, return_tensors="pt")

    # Generate the question from the model
    with torch.no_grad():
        question_outputs = model.generate(input_ids, 
                                          max_length=512,  # Reduced max_length from 1024 to 512
                                          num_return_sequences=1, 
                                          pad_token_id=tokenizer.eos_token_id,
                                          temperature=temperature,
                                          top_k=top_k,
                                          top_p=top_p)

    # Decode the generated question
    question = tokenizer.decode(question_outputs[0], skip_special_tokens=True)

    # Combine context and question to answer
    context_question = context + " " + question

    # Tokenize again to generate the answer
    input_ids_answer = tokenizer.encode(context_question, return_tensors="pt")

    # Generate the answer from the model
    with torch.no_grad():
        answer_outputs = model.generate(input_ids_answer, 
                                        max_length=1024,  # Reduced max_length from 2000 to 1024
                                        num_return_sequences=1, 
                                        pad_token_id=tokenizer.eos_token_id,
                                        temperature=temperature,
                                        top_k=top_k,
                                        top_p=top_p)

    # Decode the generated answer
    answer = tokenizer.decode(answer_outputs[0], skip_special_tokens=True)

    return question, answer

def main():
    context = "Neutrinos & Non-proliferation in Europe Michel Cribier* APC, Paris CEA/Saclay, DAPNIA/SPP The International Atomic Energy Agency (IAEA) is the United Nations agency in charge of the development of peaceful use of atomic energy. In particular IAEA is the verification authority of the Treaty on the Non-Proliferation of Nuclear Weapons (NPT). To do that jobs inspections of civil nuclear installations and related facilities under safeguards agreements are made in more than 140 states. IAEA uses many different tools for these verifications, like neutron monitor, gamma spectroscopy, but also bookeeping of the isotopic composition at the fuel element level before and after their use in the nuclear power station. In particular it verifie that weapon-origin and other fissile materials that Russia and USA have released from their defense programmes are used for civil application. The existence of an antineutrino signal sensitive to the power and to the isotopic composition of a reactor core, as first proposed by Mikaelian et al. and as demonstrated by the Bugey and Rovno experiments, , could provide a means to address certain safeguards applications. Thus the IAEA recently ask members states to make a feasibility study to determine whether antineutrino detection methods might provide practical safeguards tools for selected applications. If this method proves to be useful, IAEA has the power to decide that any new nuclear power plants built has to include an antineutrino monitor. Within the Double Chooz collaboration, an experiment mainly devoted to study the fundamental properties of neutrinos, we thought that we were in a good position to evaluate the interest of using antineutrino detection to remotely monitor nuclear power station. This effort in Europe, supplemented by the US effort , will constitute the basic answer to IAEA of the neutrino community. * On behalf of a collective work by S. Cormon, M. Fallot, H. Faust, T. Lasserre, A. Letourneau, D. Lhuillier, V. Sinev from DAPNIA, Subatech and ILL. Figure 1 : The statistical distribution of the fission products resulting from the fission of the most important fissile nuclei 235U and 239Pu shows two humps, one centered around masses 100 and the other one centered around 135."
    # Generate the question and answer from the context
    question, answer = generate_question_and_answer(context)
    
    # Print the generated question and answer
    print(f"Context: {context}")
    print(f"Generated Question: {question}")
    print(f"Generated Answer: {answer}")

if __name__ == "__main__":
    main()



