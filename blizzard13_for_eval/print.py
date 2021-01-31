import os
def print_dir(dir):
    conts = []
    names = []
    out = "outfile.txt"
    for f in os.listdir(dir):
        name = f[:-4]
        with open(os.path.join(dir, f), "r") as fl:
            l = fl.readline()
            conts.append(l)
            names.append(name)

    with open(out, "w+") as fl:
        for n, c in zip(names, conts):
            fl.write(n + " " + c + "\n")


if __name__ == '__main__':
    dir = "/home/rosen/Project/ser_model/blizzard13_for_eval/jane_eyre"
    print_dir(dir)