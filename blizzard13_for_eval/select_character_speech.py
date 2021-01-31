import os

def pickup_character_speech(chrct_speech_f, splitted_dir):
    """
    pick up character speech name from splitted dir.

    This function is used to prepare candidate emotional speech sentences.

    Parameter
        chrct_speech_f: each sentence in 1 line
                            example: "I told you don't do it"
        splitted_dir  : all sentences, including character and non-character speech, in splitted file
    return:
        chrct_speech_n: speech name in 1 line
                            example: "CB-JE-20-318"

    """
    chrct_speech_n = []
    return chrct_speech_n


if __name__ == '__main__':
    chrct_speech_f, splitted_dir = "", ""
    chrct_speech_n = pickup_character_speech(chrct_speech_f, splitted_dir)

    splitted_dir_charct = splitted_dir + "_charct"
    for n in chrct_speech_n:
        os.system("cp {} {}".format(os.path.join(splitted_dir, n), splitted_dir_charct))