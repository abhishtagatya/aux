from pydub import AudioSegment

M4A_EXT = ".m4a"
WAV_EXT = ".wav"


def m4a_to_wav(m4a_filename: str, wav_filename: str):
    """
    Convert M4A to WAV Format

    :param m4a_filename: M4A Filename
    :param wav_filename: WAV Filename
    :return:
    """
    sound = AudioSegment.from_file(m4a_filename, format='m4a')
    sound.export(wav_filename, format='wav')
    return
