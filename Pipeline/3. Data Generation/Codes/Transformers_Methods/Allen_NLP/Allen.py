# %%
# How to Download AllenNLP - https://github.com/allenai/allennlp?tab=readme-ov-file
# conda info --envs
# conda activate allennlp_env
# git clone https://github.com/allenai/allennlp.git
# cd allennlp
# pip install -r requirements.txt 
# pip install allennlp
# !pip install spacy "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz"

# %%
passage = """
In spite of the uncertainty mentioned previously, we see that the most energetic part offers the best possibility to disentangle fission from 235U and 239Pu. The comparison between the cumulative numbers of antineutrinos as a function of antineutrino energy detected at low vs. high energy is an efficient observable to distinguish pure 235U and 239Pu. IAEA seeks also monitoring large spent-fuel elements. For this application, the likelihood is that antineutrino detectors could only make measurements on large quantities of beta-emitters, e.g., several cores of spent fuel. In the time of the experiment the discharge of parts of the core will happen and the Double-Chooz experiment will quantify the sensitivity of such monitoring. More generally the techniques developed for the detection of antineutrinos could be applied for the monitoring of nuclear activities at the level of a country. Hence a KamLAND type detector deeply submerged off the coast of the country, would offer the sensitivity to detect a new underground reactor located at several hundreds of kilometers. All these common efforts toward more reliable techniques, remotely operated detectors, not to mention undersea techniques will automatically benefit to both fields, safeguard and geo-neutrinos.

"""

# %%
#Initialize the predictor
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz")


# %%
result=predictor.predict(
  passage=passage,
  question= "The most energetic part offers the best chance to disentangle fission?"
)
result['best_span_str']

# %%
result=predictor.predict(
  passage=passage,
  question= "How can the possibility to use antineutrinos for power monitoring be evaluated?"
)
result['best_span_str']

# %%
result=predictor.predict(
  passage=passage,
  question= "How can we simulate the evoluti of nuclei?"
)
result['best_span_str']

# %%
#URL
model_url = "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-2022.02.10.tar.gz"

# Predictor 
predictor = Predictor.from_path(model_url)

question = "What is the capital of France?"
passage = "Paris is the capital and largest city of France."
result = predictor.predict(passage=passage, question=question)
#Result
print(result)


# %% [markdown]
# # -------------------------------------------

# %%
from allennlp.predictors.predictor import Predictor
# predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz")

# %%
result=predictor.predict(
  passage=passage,
  question= "What would a KamLAND type detector detect?"
  # "how are union territoris managed?"
)
result['best_span_str']

# %%
result=predictor.predict(
  passage=passage,
  question= "What is the name of the neutron source used in Europe?"
  # "how are union territoris managed?"
)
result['best_span_str']

# %%
result=predictor.predict(
  passage=passage,
  question= "How can the possibility to use antineutrinos for power monitoring be evaluated?"
  # "how are union territoris managed?"
)
result['best_span_str']


