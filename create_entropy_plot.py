import matplotlib.pyplot as plt
import numpy as np


COLORS = ['#800000', '#e6194B', '#9A6324', '#f58231', '#808000', '#ffe119', '#bfef45', '#3cb44b',
          '#aaffc3', '#469990', '#42d4f4', '#000075', '#4363d8', '#911eb4', '#e6beff', '#f032e6',
          '#a8687a', '#000000']


def produce_plots(create_plot, create_critical_plot, directory, size, act_lev, act_prob, letter=''):
    gs = []
    eps = []
    ents = []
    if letter: letter = '_' + letter
    with open(f'{directory}/grid_{size}_activity_{act_lev}_prob_{act_prob}{letter}.txt') as f:
        while True:
            a = f.readline()
            if not a: break
            g, ep, ent = a.split(' ')
            g, ep, ent = float(g), float(ep), float(ent[:7])
            gs.append(g)
            eps.append(ep)
            ents.append(ent)

    init_g = gs[0]
    current_eps = []
    current_ents = []
    max_ent = {}
    index = 0

    for i in range(len(gs)):
        g = gs[i]
        if g == init_g:
            current_eps.append(eps[i])
            current_ents.append(ents[i])
            if i == len(gs) - 1:
                if create_plot:
                    plt.scatter(current_eps, current_ents, label=round(init_g, 4), c=COLORS[index % len(COLORS)])
                ep_at_max = current_eps[int(np.argmax(current_ents))]
                max_ent[init_g] = ep_at_max
                index += 1
        else:
            if create_plot:
                plt.scatter(current_eps, current_ents, label=round(init_g, 4), c=COLORS[index % len(COLORS)])
            ep_at_max = current_eps[int(np.argmax(current_ents))]
            max_ent[init_g] = ep_at_max
            index += 1
            init_g = gs[i]
            current_eps = [eps[i]]
            current_ents = [ents[i]]
    if create_plot:
        plt.legend()
        plt.ylabel('Entropy')
        plt.xlabel('Entrance Probability')
        plt.title(f'Grid size {size}, activity level {act_lev}')
        plt.show()
    if create_critical_plot:
        i = 0
        for g in max_ent:
            plt.scatter(g, max_ent[g], label=round(g, 4), c=COLORS[i % len(COLORS)])
            i += 1
        plt.legend()
        plt.xlabel('g')
        plt.ylim(0, 1)
        plt.ylabel('Critical Entrance Probability')
        plt.title(f'Critical ep, grid size {size}, activity level {act_lev}')
        plt.show()


def main():
    size = 50
    act_lev = 1e-15
    activation_prob = 0.0001
    directory = 'results'
    produce_plots(create_plot=True,
                  create_critical_plot=True,
                  directory=directory,
                  size=size,
                  act_lev=act_lev,
                  act_prob=activation_prob,
                  letter='a'
                  )


if __name__ == '__main__':
    main()
