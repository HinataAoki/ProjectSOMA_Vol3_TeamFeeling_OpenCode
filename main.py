import argparse
from Record import record
from MovieToLabel import utils as ml_utils
from SpeechRecognition import sr_utils
from googletrans import Translator

parser = argparse.ArgumentParser(description='ReCreation GAN') 
parser.add_argument('-mp', '--movie_path', help='path of movie', default='temp.mp4')
parser.add_argument('-vp', '--voice_path', help='path of movie', default='temp.wav')
args = parser.parse_args()

movie_path = args.movie_path
voice_path = args.voice_path

# 動画を作成する
record.make_record(path=movie_path)

# 動画から状況を説明するテキストを生成
generated_text_from_movie = ml_utils.make_text_from_movie(movie_path)

# Speech to Text
generated_text_from_speech = sr_utils.voice_to_text(movie_path, voice_path)

# Translate to English
translator = Translator()
translated_speech_text = translator.translate(generated_text_from_speech, dest="ja");

# テキストから画像を生成
#generate_image_from_text(generated_text)