
import os

def extract_char_name(chrct_sentnc, all_name_sent, name_pred):
    chrct_ls = []
    with open(chrct_sentnc, "r") as inf:
        for l in inf:
            chrct_ls.append(l)

    name_sent_dic = {}
    with open(all_name_sent, "r") as inf:
        for l in inf:
            name, sentence = l.split(" ", 1)
            name_sent_dic[sentence] = name

    # character_names
    chrct_name_ls = []
    for s in chrct_ls:
        if s in name_sent_dic.keys():
            chrct_name_ls.append(name_sent_dic[s])
    with open("chrct_name.txt", "w") as ouf:
        for i in sorted(chrct_name_ls):
            ouf.write(i+"\n")

    # character names_

    # character_name pred
    name_pred_dic = {}
    with open(name_pred, "r") as inf:
        for l in inf:
            name, pred = l.split(" ")
            name_pred_dic[name] = pred
    chrct_name_pred = []
    for c_n in chrct_name_ls:
        if c_n in name_pred_dic.keys():
            chrct_name_pred.append(c_n + " " + name_pred_dic[c_n])
    with open("chrct_name_pred.csv", "w") as ouf:
        for c_n_p in sorted(chrct_name_pred):
            ouf.write(c_n_p)

    # copy character wav to dir
    chcter_dir = "jane_eyer_character_dir"
    source_dir = "/home/Data/blizzard2013_part/wav1/jane_eyre"
    if os.path.isdir(chcter_dir):
        os.system("rm -rf {}".format(chcter_dir))
    else:
        os.system("mkdir {}".format(chcter_dir))
    for a in chrct_name_ls:
        chcter_audio = os.path.join(source_dir, a+".wav")
        os.system("cp {} {}".format(chcter_audio, chcter_dir))





if __name__ == '__main__':
    chrct_sentenc = "/home/rosen/Project/ser_model/blizzard13_for_eval/char_sentence.txt"
    all_name_sent = "/home/rosen/Project/ser_model/blizzard13_for_eval/outfile.txt"
    name_pred = "/home/rosen/Project/ser_model/blizzard13_for_eval/emo_feats_elbs.csv"
    extract_char_name(chrct_sentenc, all_name_sent, name_pred)


