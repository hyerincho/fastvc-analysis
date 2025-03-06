import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt


def matplotlib_settings():
    """
    Makes some modifications to the default matplotlib settings.
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({"font.size": 18})  # , "text.usetex": True})
