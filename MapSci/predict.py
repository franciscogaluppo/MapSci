import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def gaussian_kernel(ax, x, y, xlab, ylab, title, cmap='coolwarm'):
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
    ax.plot([0, xmax], [0, xmax], "--", color="darkgrey")
    
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    ax.imshow(np.rot90(f), cmap=cmap,
        extent=[xmin, xmax, ymin, ymax], aspect='auto')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    plt.title(title)


def predict_all(entity, spaces, idx, transition, future):
    """
    """
    if not isinstance(spaces, list):
        spaces = [spaces]

    singles = 0
    auc = [list() for x in spaces]
    
    for s in entity.set:
        pred = [entity.predict(s, x, transition) for x in spaces]
        prob = [[x[0] for x in p] for p in pred]
        true = [[idx[x[1]] in future._U[u][s] for x in p] for p in pred]

        try:
            for i in range(len(prob)):
                auc[i].append(roc_auc_score(true[i], prob[i]))
        except:
            singles += 1

    tot = len(entity.set)
    print("{} out of {} scores couldn't be computed.".format(singles,tot))
    return auc 


def __adj_val(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, min(vals))

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, min(vals), q1)
    return lower_adjacent_value, upper_adjacent_value


def plot_comp(auc, entity):
    """
    """
    X, areas = entity.info()

    # Plots
    plt.rcParams["figure.figsize"] = (18,18)

    #plt.violinplot(auc, points=60, widths=0.7,
    #    showextrema=True, showmedians=True, bw_method=0.5)
    #plt.title("AUC ROC curve distributions")
    #plt.ylabel('AUC ROC')
    #plt.xticks([1, 2], ['1', '2'])

    ax = plt.subplot(3,3,1)
    ax.set_title("AUC ROC curve distributions")
    parts = ax.violinplot(auc, showmeans=False, showmedians=True,
        showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    mean = np.mean(auc, axis=1)
    q1, q3 = np.percentile(auc, [25, 75], axis=1)
    whiskers = np.array([__adj_val(vals, qq1, qq3) for vals, qq1, qq3 \
        in zip(auc, q1, q3)])
    wMin, wMax = whiskers[:,0], whiskers[:,1]

    inds = np.arange(1, len(q1) + 1)
    ax.scatter(inds, mean, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, q1, q3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, wMin, wMax, color='k', linestyle='-', lw=1)
    set_axis_style(ax, ['1','2'])
    
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


def summary(auc):
    """
    """
    wins = [0,0,0]
    wins_over = [0,0,0]

    for i in range(auc[0]):
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
