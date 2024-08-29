import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_data(csv_file):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    - csv_file (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(csv_file)
    return data

def data_processing(filepath, method):
    data = load_data(filepath)
    d = data.iloc[-1,:]
    c = pd.DataFrame([obj[i], method], index=['shape', "method"])
    d_tcn = pd.DataFrame([obj[i], method], index=['shape', "method"])
    dd = pd.concat([d,c], axis=0)
    return dd


if __name__ == "__main__":
    # Specify the path to your CSV file
    date = "01-19"
    date_abration = "02-13"
    date_tcn = "02-07"
    date_student = "01-28"
    date_genralization = "02-05"
    obj = ["circle", "square", "triangle", "rectangle"]
    angle = 5
    # fig, axes = plt.subplots(1, 4, figsize=(10, 10))
    exp_data = []
    exp_data_noangle = []
    exp_data_nohole = []
    exp_data_nostiffness = []
    exp_data_noprivilege = []
    
    for i in range(1):
        for seed in range(10):
            name = date + "-" + obj[i] + "-peg-hole-pegangle-seed-10-" + str(angle) +"-" + str(seed)
            name_noangle = date_abration + "-" + obj[i] + "-peg-hole-pegangle-seed-10-" + str(0) +"-" + str(seed)
            name_nohole = date_abration + "-" + obj[i] + "-peg-hole-pegangle-seed-0-" + str(5) +"-" + str(seed)
            name_nostiffness = date_abration + "-" + obj[i] + "-fixedstiffness-peg-hole-pegangle-seed-10-" + str(5) +"-" + str(seed)
            name_noprivilege = date + "-" + obj[i] + "-noalignment-peg-hole-pegangle-seed-10-" + str(5) +"-" + str(seed)    
            csv_file_path = "./learning_progress/" + name + "/" + name + "_success_rates.csv"
            csv_file_path_noangle = "./learning_progress/" + name_noangle + "/" + name_noangle + "_success_rates.csv"
            csv_file_path_nohole = "./learning_progress/" + name_nohole + "/" + name_nohole + "_success_rates.csv"
            csv_file_path_nostiffness = "./learning_progress/" + name_nostiffness + "/" + name_nostiffness + "_success_rates.csv"
            csv_file_path_noprivilege = "./learning_progress/" + name_noprivilege + "/" + name_noprivilege + "_success_rates.csv"           
            dd = data_processing(csv_file_path, "all")
            dd_noangle = data_processing(csv_file_path_noangle, "fixed angle")
            dd_nohole = data_processing(csv_file_path_nohole, "fixed hole")
            dd_nostiffness = data_processing(csv_file_path_nostiffness, "fixed stiffness")
            dd_noprivilege = data_processing(csv_file_path_noprivilege, "no alignment")

            if seed == 0:
                e = dd.T
                e_noangle = dd_noangle.T
                e_nohole = dd_nohole.T
                e_nostiffness = dd_nostiffness.T
                e_noprivilege = dd_noprivilege.T
            else:
                e = pd.concat([e, dd.T], axis=0, join='inner')
                e_noangle = pd.concat([e_noangle, dd_noangle.T], axis=0, join='inner')
                e_nohole = pd.concat([e_nohole, dd_nohole.T], axis=0, join='inner')
                e_nostiffness = pd.concat([e_nostiffness, dd_nostiffness.T], axis=0, join='inner')
                e_noprivilege = pd.concat([e_noprivilege, dd_noprivilege.T], axis=0, join='inner')
        if i ==0:
            exp_data = e
            exp_data_noangle = e_noangle
            exp_data_nohole = e_nohole
            exp_data_nostiffness = e_nostiffness
            exp_data_noprivilege = e_noprivilege
        else:
            exp_data = pd.concat([exp_data, e], axis=0, join='inner')
            exp_data_noangle = pd.concat([exp_data_noangle, e_noangle], axis=0, join='inner')
            exp_data_nohole = pd.concat([exp_data_nohole, e_nohole], axis=0, join='inner')
            exp_data_nostiffness = pd.concat([exp_data_nostiffness, e_nostiffness], axis=0, join='inner')
            exp_data_noprivilege = pd.concat([exp_data_noprivilege, e_noprivilege], axis=0, join='inner')

    x_column = "method"
    y_column = "sum success"
    # y_column = "termination step"
    results = pd.concat([exp_data, exp_data_noprivilege, exp_data_noangle, exp_data_nohole, exp_data_nostiffness], axis=0, join='inner')
    plt.xlabel("teacher policy",fontsize=20)
    plt.ylabel("success rate [%]", fontsize=20)
    palette=sns.color_palette(['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499']) 
    sns.set_palette(palette)
    # Plot the experimental results
    # sns.set(style="whitegrid")  # Set Seaborn style
    # ax=sns.barplot(x=x_column, y=y_column, hue="method", data=results, errorbar=("sd", 1),edgecolor="black",
    # err_kws={'linewidth': 2.5},
    # capsize = 0.05)
    ax=sns.boxplot(x=x_column, y=y_column, hue="method", data=results, linewidth=3)
    sns.stripplot(x=x_column, y=y_column, hue="method", data=results, size=9, edgecolor="black",linewidth=1)
    handles, labels = ax.get_legend_handles_labels()
    #sns.set(font_scale=10)
    plt.tick_params(axis='both', which='major', labelsize=18)
    # ax.legend(handles[:5], labels[:5], loc='upper right')
    # plt.tight_layout()
    plt.savefig("results"+ "hole" +str(10)+ "angle" + str(angle)+"_success.png")
    plt.show()