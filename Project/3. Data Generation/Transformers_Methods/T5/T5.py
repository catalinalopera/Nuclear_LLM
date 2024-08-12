# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
from transformers import T5ForConditionalGeneration, T5Tokenizer

question = "What is the statistical distribution of fission products resulting from the fission?"
context = "Neutrinos & Non-proliferation in Europe Michel Cribier* APC, Paris CEA/Saclay, DAPNIA/SPP The International Atomic Energy Agency (IAEA) is the United Nations agency in charge of the development of peaceful use of atomic energy. In particular IAEA is the verification authority of the Treaty on the Non-Proliferation of Nuclear Weapons (NPT). To do that jobs inspections of civil nuclear installations and related facilities under safeguards agreements are made in more than 140 states. IAEA uses many different tools for these verifications, like neutron monitor, gamma spectroscopy, but also bookeeping of the isotopic composition at the fuel element level before and after their use in the nuclear power station. In particular it verifie that weapon-origin and other fissile materials that Russia and USA have released from their defense programmes are used for civil application. The existence of an antineutrino signal sensitive to the power and to the isotopic composition of a reactor core, as first proposed by Mikaelian et al. and as demonstrated by the Bugey and Rovno experiments, , could provide a means to address certain safeguards applications. Thus the IAEA recently ask members states to make a feasibility study to determine whether antineutrino detection methods might provide practical safeguards tools for selected applications. If this method proves to be useful, IAEA has the power to decide that any new nuclear power plants built has to include an antineutrino monitor. Within the Double Chooz collaboration, an experiment mainly devoted to study the fundamental properties of neutrinos, we thought that we were in a good position to evaluate the interest of using antineutrino detection to remotely monitor nuclear power station. This effort in Europe, supplemented by the US effort , will constitute the basic answer to IAEA of the neutrino community. * On behalf of a collective work by S. Cormon, M. Fallot, H. Faust, T. Lasserre, A. Letourneau, D. Lhuillier, V. Sinev from DAPNIA, Subatech and ILL. Figure 1 : The statistical distribution of the fission products resulting from the fission of the most important fissile nuclei 235U and 239Pu shows two humps, one centered around masses 100 and the other one centered around 135."

# %%
model_name = 't5-small'  
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# %%
context = " ".join(context.split()[:512])

input_text = f"question: {question} context: {context}"
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

outputs = model.generate(**inputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("question:", question)
print("vontext:", context)
print("answer:", answer)

# %%


# %%



