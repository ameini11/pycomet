import os


def replace(file, patterns, replaces):
    file_old = open(file, 'r')
    file_new = open(file+'_new', 'w')
    for line in file_old:
        if any([p in line for p in patterns]):
            for i, p in enumerate(patterns):
                if p in line:
                    line_new = replaces[i]
                    break
        else:
            line_new = line
        file_new.write(line_new)
    file_new.close()
    file_old.close()
    os.rename(file+'_new', file)