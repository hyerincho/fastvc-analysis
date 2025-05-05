from plot_utils import *
from matplotlib_settings import *


def compare_eta_phib(dirtags, labels=None, colors=None, average_factor=1.2):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if colors is None:
        colors = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtags)))  # colors for each runs
    if labels is None:
        labels = dirtags

    for i, dirtag in enumerate(dirtags):
        pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
        with open(pkl, "rb") as openFile:
            D = pickle.load(openFile)

        eta_mean = extractQuantity(D, "eta", average_factor)
        phib_mean = extractQuantity(D, "phib", average_factor)
        a = get_spin(D)

        ax.plot(phib_mean, eta_mean, marker=".", color=colors[i], label=labels[i])

    xlim = ax.get_xlim()
    phib_xaxis = np.linspace(xlim[0], xlim[1], 10)
    ax.plot(phib_xaxis, eta_BZ6(a, phib_xaxis, 0.03))
    ax.legend()  # fontsize=7)
    ax.set_xlabel(r"$\phi_b$")
    ax.set_ylabel(r"$\eta$")
    fig.tight_layout()
    output = "../plots/compare_eta_phib.png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)


def compare_kappa(dirtags):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtags)))  # colors for each runs

    for i, dirtag in enumerate(dirtags):
        pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
        with open(pkl, "rb") as openFile:
            D = pickle.load(openFile)

        eta_mean = extractQuantity(D, "eta")
        phib_mean = extractQuantity(D, "phib")
        a = get_spin(D)

        kappa = eta_mean / eta_BZ6(a, phib_mean, 1)

        ax.plot(phib_mean, kappa, marker=".", color=colors[i], label=dirtag)

    ax.legend(fontsize=7)
    ax.axhline(0.044)
    ax.axhline(0.053)
    ax.set_xlabel(r"$\phi_b$")
    ax.set_ylabel(r"$\kappa$")
    fig.tight_layout()
    output = "../plots/compare_kappa.png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)


if __name__ == "__main__":
    dirtags = ["032125_torus_tegan", "032425_torus", "032525_torus_safe_floors_normal", "032625_torus_safe_floors2", "032725_torus_safe_floors3", "032825_torus_safe_floors4", "032825_torus_safe_floors_nofofc_sigmamax", "040125_torus_reconnect_sigma", "040225_torus_reconnect_sigma_Tmax", "040325_torus_noehbuffer"]
    dirtags = ["031125_a0.9_cap_correctly", "032125_n4a0.9_toriilike", "042125_n4_a0.9_bondi_jks2", "041625_n4_a0.9_toriilike_jks2_smth2_reconnect"]
    dirtags = [
        # "030325_a0.9_oz_128",
        "delta/030525_a0.9_oz",
        "042125_a0.9_oz_jks",
        "031125_a0.9_cap_correctly",
        "042125_n4_a0.9_bondi_jks2",
        "042225_n4_a0.9_bondi_jks2_clearangle",
        "delta/032025_a0.9_oz_clearangle",
        "032125_n4a0.9_toriilike",
        "041625_n4_a0.9_toriilike_jks2_smth2_reconnect",
        "042225_n4_a0.9_retrograde",
        # "042325_n4_a0.9_tl_uphi0",
        # "042225_n4_a0.9_toriilike_jks2_nocap",
        # "042225_n4_a0.9_bondi_jks2_nocap"
        "040325_n4_a0.9_torrilike_nocap",
        "042325_a0.9_rB2e3_bondi",
        # "041825_a0.9_rB2e5_jks2_smth2",
        "042325_a0.9_rB2e5_bondi",
    ]
    labels = ["oz", "oz_jks", "mz", "mz_jks", "mz_jks_ca", "oz_tl", "mz_tl", "mz_tl_jks", "mz_tl_-.9", "mz_tl_nocap", "2e3", "2e5"]  # "mz_tl_0",
    colors = ["black", "tab:blue", "c", "g", "gray", "r", "tab:orange", "m", "y", "pink", "peru", "orange"]  # colors for each runs
    compare_eta_phib(dirtags, labels, colors, 1.5)  # 1.1)
    # compare_kappa(dirtags)

    # print(eta_BZ6(0.9375, 56.37, 0.044))
    # print(eta_BZ6(0.9375, 44.8, 0.054))
    # print(eta_BZ6(0.9, 49.695, 0.044))
