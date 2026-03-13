import MetricMathStreamline as mm
import MainLoopStreamline as ml
import numpy as np
import matplotlib.pyplot as plt

goof = ml.load_index()
ins_refs = [name for name, val in goof.items() if "Paper" in val["Label"]]
data_stuff, cool_stuff = {}, {}
for name in ins_refs:
    data = ml.load_emri_data(name)
    lab = data["name"].split()[0]
    if lab not in data_stuff.keys():
        print(lab)
        data_stuff[lab] = [[], [], []]
        cool_stuff[lab] = [data["spin"], data["energy"][0], data["phi_momentum"][0], data["carter"][0]]
    data_stuff[lab][0].append(len(data["p"]))
    data_stuff[lab][1].extend(data["p"])
    data_stuff[lab][2].extend(data["e"])
    del data

for key in data_stuff.keys():
    plt.figure()
    print(key, "!!!!")
    seps = np.cumsum(data_stuff[key][0]) 
    a, E, L, C = cool_stuff[key] 
    test1 = mm.glamp_2002(a, 1e-7, cons=[E, L, C]) 
    test2 = mm.gair_glamp_2006(a, 1e-7, cons=[E, L, C]) 
    print(data_stuff[key][0]) 
    print(len(data_stuff[key][1])) 
    for i in range(len(seps)): 
        st_ix = 0 if i == 0 else seps[i-1] 
        end_ix = seps[i] if i < len(seps) -1 else seps[-1] + 10 
        print(st_ix, end_ix) 
        plt.plot(data_stuff[key][1][st_ix:end_ix], data_stuff[key][2][st_ix:end_ix]) 
        #plt.plot(data_stuff[key], es[key], label=len()) 
    xlim = plt.xlim() 
    ylim = plt.ylim() 
    plt.plot(test1["p"], test1["e"], linestyle="--", color="gray") 
    plt.plot(test2["p"], test2["e"], linestyle="--", color="k") 
    p, e, r = mm.get_sep_cosi(a, L/np.sqrt(L**2 + C)) 
    plt.plot(p, e, linestyle=":", color="k") 
    plt.xlim(xlim) 
    plt.ylim(ylim) 
    plt.title(key)
    plt.show(block = False)

input("Done? Press enter to close plots. ")