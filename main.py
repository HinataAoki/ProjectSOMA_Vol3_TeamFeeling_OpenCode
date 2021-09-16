import argparse
from Record import record
from MovieToLabel import utils as ml_utils
from SpeechRecognition import sr_utils
from googletrans import Translator

parser = argparse.ArgumentParser(description='ReCreation GAN') 
parser.add_argument('-mp', '--movie_path', help='path of movie', default='temp.mp4')
parser.add_argument('-vp', '--voice_path', help='path of voice', default='temp.wav')
parser.add_argument('-th', '--threthold', help='threthold', default='0.5')
parser.add_argument('--img_dir', help='path of folder saving temp imgs', default='temp')
parser.add_argument('--img_name', help='name of saving temp imgs', default='temp')
args = parser.parse_args()

movie_path = args.movie_path
voice_path = args.voice_path
img_dir = args.img_dir
img_name = args.img_name
threthold = float(args.threthold)

# 動画を作成する
record.make_record(path=movie_path)

# 動画から状況を説明するテキストを生成
generated_text_from_movie = ml_utils.movie_to_text(movie_path, img_dir="temp", img_name="temp", threthold=0.5)

# Speech to Text
generated_text_from_speech = sr_utils.voice_to_text(movie_path, voice_path)

# Translate to English
translator = Translator()
translated_speech_text = translator.translate(generated_text_from_speech, dest="ja")

# テキストから画像を生成
#generate_image_from_text(generated_text)