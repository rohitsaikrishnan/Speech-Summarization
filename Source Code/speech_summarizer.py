from deepspeech import Model
from pydub import AudioSegment
from summarizer import SummaryExtractor
import spacy
from rouge import Rouge
import pytorch
import numpy as np
from scipy.io.wavfile import write
import os


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)

def main():
    """
    Converting Mp3 file to WAV file for all the audio files
    """
    with open("audio_files.txt") as audio_files:
        audio = audio_files.read().splitlines()
    for audio_file in audio:
        src = audio_file + ".mp3"
        dst = audio_file + ".wav"
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")

    for audio_file in audio:
        """
        Speech - to text model
        """
        ds = Model('deepspeech-0.7.0-models.pbmm')
        desired_sample_rate = ds.sampleRate()
        fin = wave.open(args.audio, 'rb')
        fs_orig = fin.getframerate()
        if fs_orig != desired_sample_rate:
            print(
                'Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(
                    fs_orig, desired_sample_rate), file=sys.stderr)
            fs_new, audio = convert_samplerate(args.audio, desired_sample_rate)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

        audio_length = fin.getnframes() * (1 / fs_orig)
        fin.close()
        audio_file_path = audio_file + ".wav"
        speech_to_text = ds.stt(audio_file_path)
        """
        Sentence Tokenization
        """
        nlp = spacy.load('en_core_web_md')
        text_sentences = nlp(text)
        for sentences in text_sentences.sents:
            sentence=sentences.text+". "
            sentence_text=sentence_text + sentence
        final_Speech_to_text = sentence_text
        """
        BERT and Clustering based Techniques for Text Summarization
        """
        model = SummaryExtractor()
        summarized_text = model(final_Speech_to_text)
        """
        Fetching Gold Standard summarized text
        """
        hypothesis_source = "gold_standard" + "//" + audio_file + ".txt"
        with open(hypothesis_source) as gold_standard:
            hypothesis = gold_standard.read()
        """
        ROUGE SCORES Evaluation Metric
        """
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, summarized_text)
        print(scores)
        """
        Summarized Text-to-Speech Model
        """
        tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
        tacotron2 = tacotron2.to('cuda')
        tacotron2.eval()
        waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to('cuda')
        waveglow.eval()
        # preprocessing
        sequence = np.array(tacotron2.text_to_sequence(summarized_text, ['english_cleaners']))[None, :]
        sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

        # run the models
        with torch.no_grad():
            _, mel, _, _ = tacotron2.infer(sequence)
            audio = waveglow.infer(mel)
        audio_numpy = audio[0].data.cpu().numpy()
        rate = 22050
        destination_add = "summarized_speeches//" + audio_file +".wav"
        write(destination_add, rate, audio_numpy)
        """
        Convert From WAV to MP3
        """
        src = "summarized_speeches//" + audio_file +".wav"
        dst = "summarized_speeches//" + audio_file +".mp3"
        sound = AudioSegment.from_wav(src)
        sound.export(dst, format="mp3")
        os.remove(src)
    """
    Speech Summarization of 50 speeches Done
    """


