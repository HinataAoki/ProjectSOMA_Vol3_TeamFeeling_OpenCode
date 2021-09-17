import gc
import ffmpeg
import speech_recognition as sr

def voice_to_text(movie_path,voice_path,text_path=None):
  #mp4からwavへ変換
  stream = ffmpeg.input(movie_path)
  stream = ffmpeg.output(stream, voice_path)
  ffmpeg.run(stream)
  #SpeachRecognition
  r = sr.Recognizer()
  with sr.AudioFile(voice_path) as source:
    audio = r.record(source)
    text = r.recognize_google(audio, language="ja-JP").replace(" ", "\n")
    if text_path is not None:
      open_text = open(text_path, "x", encoding="utf_8")
      open_text.write(text)
      open_text.close()
    del stream, movie_path, voice_path, text_path, source, audio
    gc.collect()
    return text