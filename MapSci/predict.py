import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def gaussian(ax, x, y, xlab, ylab, title, cmap='coolwarm', iden=False):
    """
    """
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    if iden:
        ax.plot([0, xmax], [0, xmax], "--", color="darkgrey")
    
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    ax.imshow(np.rot90(f), cmap=cmap,
        extent=[xmin, xmax, ymin, ymax], aspect='auto')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    plt.title(title)


def predict_futures(entity, space, idx, transition, futures):
    """
    """
    if not isinstance(futures, list):
        futures = [futures]

    singles = 0
    auc = [list() for x in futures]
    u =  transition != 'inactive-active'
    n = len(futures)
    
    for s in entity.set:
        pred = entity.predict(s, space, transition)
        prob = [x[0] for x in pred]
        true = [[idx[x[1]] in f._U[u][s] for x in pred] for f in futures]

        try:
            scores = list()
            for i in range(n):
                scores.append(roc_auc_score(true[i], prob))
            for i in range(n):
                auc[i].append(scores[i])
        except:
            singles += 1

    tot = len(entity.set)
    print("{} out of {} scores couldn't be computed.".format(singles,tot))
    return auc


def predict_all(entity, spaces, idx, transition, future):
    """
    """
    if not isinstance(spaces, list):
        spaces = [spaces]

    singles = 0
    computed = list()
    auc = [list() for x in spaces]
    u =  transition != 'inactive-active'
    n = len(spaces)
    
    for s in entity.set:
        pred = [entity.predict(s, x, transition) for x in spaces]
        prob = [[x[0] for x in p] for p in pred]
        true = [[idx[x[1]] in future._U[u][s] for x in p] for p in pred]

        try:
            scores = list()
            for i in range(n):
                scores.append(roc_auc_score(true[i], prob[i]))
            for i in range(n):
                auc[i].append(scores[i])
            computed.append(True)
        except:
            computed.append(False)
            singles += 1

    tot = len(entity.set)
    print("{} out of {} scores couldn't be computed.".format(singles,tot))
    return [auc, computed] 


def __adj_val(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, max(vals))

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, min(vals), q1)
    return lower_adjacent_value, upper_adjacent_value

def __axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

def plot_comp(auc, entity, computed):
    """
    """
    X, areas = entity.info()
    X = [X[i] for i in range(len(X)) if computed[i]]
    areas = [[a[i] for i in range(len(a)) if computed[i]] for a in areas]

    # Plots
    plt.rcParams["figure.figsize"] = (18,18)
    ax = plt.subplot(3,3,1)
    ax.set_title("AUC ROC curve distributions")
    parts = ax.violinplot(auc, showmeans=False, showmedians=True,
        showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    parts["cmedians"].set_edgecolor('black')

    mean = np.mean(auc, axis=1)
    q1, q3 = np.percentile(auc, [25, 75], axis=1)
    whiskers = np.array([__adj_val(vals, qq1, qq3) for vals, qq1, qq3 \
        in zip(auc, q1, q3)])
    wMin, wMax = whiskers[:,0], whiskers[:,1]

    inds = np.arange(1, len(q1) + 1)
    ax.scatter(inds, mean, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, q1, q3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, wMin, wMax, color='k', linestyle='-', lw=1)
    __axis_style(ax, ['1','2'])
    
    ax = plt.subplot(3,3,2)
    gaussian(ax, auc[0], auc[1], "1", "2", "AUC ROC comparison")
    
    ax = plt.subplot(3,3,4)
    gaussian(ax, areas[0], auc[0], "Number of active fields",
        "AUC ROC", "Active fields X AUC ROC (1)", 'PuBu')
    ax = plt.subplot(3,3,7)
    gaussian(ax, areas[0], auc[1], "Number of active fields",
        "AUC ROC", "Active fields X AUC ROC (2)", 'PuBu')
    
    ax = plt.subplot(3,3,5)
    gaussian(ax, areas[1], auc[0], "Number of developed fields",
        "AUC ROC", "Developed fields X AUC ROC (1)", 'Oranges')
    ax = plt.subplot(3,3,8)
    gaussian(ax, areas[1], auc[1], "Number of developed fields",
        "AUC ROC", "Developed fields X AUC ROC (2)", 'Oranges')
    
    ax = plt.subplot(3,3,6)
    gaussian(ax, np.log(X), auc[0], "Log-Number of papers (fractions)",
        "AUC ROC", "LogPapers X AUC ROC (1)", 'Greens')
    ax = plt.subplot(3,3,9)
    gaussian(ax, np.log(X), auc[1], "Log-Number of papers (fractions)",
        "AUC ROC", "LogPapers X AUC ROC (2)", 'Greens')
    plt.show()


def plot_violin(auc):
    """
    """
    plt.rcParams["figure.figsize"] = (2*len(auc), 7)
    ax = plt.subplot(1,1,1)
    ax.set_title("AUC ROC curve distributions")
    parts = ax.violinplot(auc, showmeans=False, showmedians=True,
        showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    parts["cmedians"].set_edgecolor('black')

    mean = np.mean(auc, axis=1)
    q1, q3 = np.percentile(auc, [25, 75], axis=1)
    whiskers = np.array([__adj_val(vals, qq1, qq3) for vals, qq1, qq3 \
        in zip(auc, q1, q3)])
    wMin, wMax = whiskers[:,0], whiskers[:,1]

    inds = np.arange(1, len(q1) + 1)
    ax.scatter(inds, mean, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, q1, q3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, wMin, wMax, color='k', linestyle='-', lw=1)
    __axis_style(ax, [str(i) for i in inds])
    ax.plot(inds, mean,color='lightgrey', linestyle='dashed', linewidth=2)
    plt.show()


def summary(auc):
    """
    """
    wins = [0,0,0]
    wins_over = [0,0,0]

    for i in range(len(auc[0])):
        if auc[0][i] > auc[1][i]:
            wins[0] += 1
            if auc[1][i] > 0.5:
                wins_over[0] += 1
        elif auc[0][i] < auc[1][i]:
            wins[1] += 1
            if auc[0][i] > 0.5:
                wins_over[1] += 1
        else:
            wins[2] += 1
            if auc[1][i] > 0.5:
                wins_over[2] += 1

    print("Anova:", st.f_oneway(*auc))
    print("Sample size:", len(auc[0]))
    print("Max value", [max(x) for x in auc])
    print("Mean value", [np.mean(x) for x in auc])
    print("Median value", [np.median(x) for x in auc])
    print("Min value", [min(x) for x in auc])
    print()


    print("Fração menor que 0.5", [sum(1 for x in a if x < 0.5) / len(a) for a in auc])
    print("Fração que foi melhor", [x / sum(wins) for x in wins[:2]])
    print("Fração que foi melhor, acima de 0.5", [x / sum(wins_over) for x in wins_over[:2]])
