import os
import uuid
from flask import Flask, flash, request, redirect, render_template, url_for
import pickle
UPLOAD_FOLDER = 'files'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


import os
from os import listdir
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import whisperx
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoConfig, Wav2Vec2Processor
from konlpy.tag import Mecab, Okt
import numpy as np
import sentence_splitter
sentencesplitter = sentence_splitter.SentenceSplitter('en')
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)


from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('vocab.pkl', 'rb') as f:
    vocab_emotion = pickle.load(f)


encode = {token: idx for idx, token in enumerate(vocab_emotion)}
decode = {idx: token for idx, token in enumerate(vocab_emotion)}

UNK = encode.get('<UNK>')
PAD = encode.get('<PAD>')
UNK, PAD

class RNNClassifier(nn.Module):
    def __init__(self, n_vocab, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding =nn.Embedding(n_vocab, embedding_dim)
        self.rnn = nn.RNN(embedding_dim,hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 8) #출력 크기 = 7: 7중 클래스 분류
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        output = self.fc(output[:,-1,:])
        output = self.sigmoid(output) #sigmoid
        return output

def pad_sequences_emotion(sequences, max_len, pad_token):
    padded = []
    seqlen = len(sequences)


    for i in range(seqlen):
        padded.append(sequences[i]) if i < seqlen else padded.append(pad_token)
    return padded



model_emotion = torch.load('model2.pth').to(device)
# %%

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        


# %%


# device = 'cpu'
audio = 'Recording.m4a'
batch_size = 16
compute_type = 'float16'
model_size = "large-v3"

LLAMA_MODEL_DIR = "nlpai-lab/KULLM3"
model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_DIR, torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_DIR)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# %%

model_audio = whisperx.load_model(model_size, device, compute_type=compute_type)
model_speech_emotion_loc = 'jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance'

config = AutoConfig.from_pretrained(model_speech_emotion_loc)
processor = Wav2Vec2Processor.from_pretrained(model_speech_emotion_loc)
sampling_rate = processor.feature_extractor.sampling_rate
model_speech = Wav2Vec2ForSpeechClassification.from_pretrained(model_speech_emotion_loc).to(device)

# %%
def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def predict_emotion(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model_speech(input_values, attention_mask=attention_mask).logits

    # print(config)
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    maxidx = np.argmax(scores)
    return scores[np.argmax(scores)] * 100, config.id2label[maxidx] 

def load_audio(audio):
    # audio_nd = whisperx.load_audio(audio)
    result = model_audio.transcribe(audio, batch_size=batch_size, language='ko')
    try:
        textresult = result['segments'][0]['text'].strip()
    except:
        textresult = ''
    return textresult



tokenjy = Okt()
def predict_emotion_jy(model, sentence, max_len, device, pad_token):
    label_to_index = {'없음':0,'기쁨':1, '놀라움':2, '사랑스러움':3, '화남':4, '슬픔':5, '두려움':6, '없음':7}
    index_to_label = {v:k for k,v in label_to_index.items()}
    model.eval()

    with torch.no_grad():
        sentence = tokenjy.morphs(sentence)
        sentence = [encode[word] if word in encode else encode['<UNK>'] for word in sentence]
        padded_sentence = pad_sequences_emotion(sentence, max_len, pad_token)
        padded_sentence = torch.tensor(padded_sentence, dtype=torch.long, device=device).unsqueeze(0)


        output = model_emotion(padded_sentence)

        predicted_class = torch.argmax(output).cpu().item()
    
    return index_to_label[predicted_class]

def load_audio(audio):
    # audio_nd = whisperx.load_audio(audio)
    result = model_audio.transcribe(audio, batch_size=batch_size, language='ko')
    try:
        textresult = result['segments'][0]['text'].strip()
    except:
        textresult = ''
    return textresult



def adviser(textresult, feelings):
    
    s=f"'{textresult}' ({feelings}) 로 말하는 사람에게 ()안의 감정을 고려하여 300자 이내로 답변해줘."

    conversation = [
        {
            "role": "system",
            "content": "나의 이름은 알파고다. 나 알파고는 'user'가 제시한 문장과 감정을 보고 문장 상황에 맞게 공감을 하며 도움이 되는 조언을 간결하게 하는 심리전문가야.",
        },
        {
            'role': 'user', 
            'content': s
        }
    ]
    
    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        # add_generation_prompt=True,
        return_tensors='pt').to("cuda")
    _ = model.generate(inputs, streamer=streamer, max_new_tokens=1000)

    response = tokenizer.decode(_[0], skip_special_tokens=True)
    try:
        response = response.split('[/INST]')[-1].strip('"')
    except Exception as e:
        print(e)
        print(response)

    return response

textresult = ''
emotion = ''
response = ''

@app.route('/')
def root():
    print(textresult, emotion, response)
    return render_template('index.html',transcript=textresult, emotion=emotion, response = response)

@app.route('/save-record', methods=['POST'])
def save_record():
    global textresult
    global emotion
    global response
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    file_name = str(uuid.uuid4()) + ".m4a"
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)
    
    textresult = load_audio(full_file_name)
    if textresult:
        # _, emotion = predict_emotion(full_file_name, sampling_rate)
        emotion = predict_emotion_jy(model_speech, textresult, 64, device, 0)
        # emotion = '없음'
        response = adviser(textresult, emotion)
    # result(textresult, emotion, response)
    # return render_template('index.html', transcript=textresult, emotion=emotion, response = response)
    response = '<br><br>'.join(sentencesplitter.split(response))

    return redirect("/", code=301)


if __name__ == '__main__':
    app.run(debug=True)



