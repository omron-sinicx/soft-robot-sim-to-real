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
    date_tcn = "02-07"
    date_student = "01-28"
    date_genralization = "02-05"
    obj = ["circle", "square", "triangle", "rectangle"]
    angle = 5
    # fig, axes = plt.subplots(1, 4, figsize=(10, 10))
    exp_data = []
    exp_data_tcn = []
    exp_data_noalign = []
    exp_data_student = []
    exp_data_student_noalign = []
    exp_data_student_generalization = []
    exp_data_student_generalization_noalign = []
    for i in range(4):
        for seed in range(10):
            name = date + "-" + obj[i] + "-peg-hole-pegangle-seed-10-" + str(angle) +"-" + str(seed) 
            name_tcn = date_tcn + "-tcn-circle-" + obj[i] + "-peg-hole-pegangle-seed-10-" + str(angle) +"-" + str(seed)
            # name_noalign = date  + "-" + obj[i] + "-noalignment-peg-hole-pegangle-seed-10-" + str(angle) +"-" + str(seed)
            # name_student = date_student + "-" + obj[i] + "-peg-hole-pegangle-seed-10-" + str(angle) +"-" + str(seed) + "-student"
            # name_student_noalign = name_noalign + "-student"
            name_generalization = date_genralization + "-circle-" + obj[i] + "-peg-hole-pegangle-seed-10-5-" + str(seed) + "-student"
            name_generalization_noalign = date_genralization + "-circle-noalignment-" + obj[i] + "-peg-hole-pegangle-seed-10-5-" + str(seed) + "-student"              
            csv_file_path = "./learning_progress/" + name + "/" + name + "_success_rates.csv"
            csv_file_path_tcn = "./learning_progress_tcn/02-07-generalization/"+ name_tcn + "_success_rates.csv"
            # csv_file_path_noalign = "./learning_progress/" + name_noalign + "/" + name_noalign + "_success_rates.csv"
            # csv_file_path_student = "./student_progress/" + name_student + "/" + name_student + "_success_rates.csv"
            # csv_file_path_student_noalign = "./student_progress/" + name_student_noalign + "/" + name_student_noalign + "_success_rates.csv"
            csv_file_path_student_generalization = "./student_progress/01-28-generalization/" + name_generalization  + "_success_rates.csv"
            csv_file_path_student_generalization_noalign = "./student_progress/01-28-generalization/" + name_generalization_noalign  + "_success_rates.csv"
            # dd = data_processing(csv_file_path, "privilege")
            dd_tcn = data_processing(csv_file_path_tcn, "TCN")
            # dd_noalign = data_processing(csv_file_path_noalign, "noalign")
            # dd_student = data_processing(csv_file_path_student, "student")
            # dd_student_noalign = data_processing(csv_file_path_student_noalign, "student noalign")
            dd_student_generalization = data_processing(csv_file_path_student_generalization, "student")
            dd_student_generalization_noalign = data_processing(csv_file_path_student_generalization_noalign, "student (no alignment)")

            if seed == 0:
                # e = dd.T
                e_tcn = dd_tcn.T
                # e_noalign = dd_noalign.T
                # e_student = dd_student.T
                # e_student_noalign = dd_student_noalign.T
                e_student_generalization = dd_student_generalization.T
                e_student_generalization_noalign = dd_student_generalization_noalign.T
                # exp_data_tcn = data_tcn
            else:
                # e = pd.concat([e, dd.T], axis=0, join='inner')
                e_tcn = pd.concat([e_tcn, dd_tcn.T], axis=0, join='inner')
                # e_noalign = pd.concat([e_noalign, dd_noalign.T], axis=0, join='inner')
                # e_student = pd.concat([e_student, dd_student.T], axis=0, join='inner')
                # e_student_noalign = pd.concat([e_student_noalign, dd_student_noalign.T], axis=0, join='inner')
                e_student_generalization = pd.concat([e_student_generalization, dd_student_generalization.T], axis=0, join='inner')
                e_student_generalization_noalign = pd.concat([e_student_generalization_noalign, dd_student_generalization_noalign.T], axis=0, join='inner')
                # exp_data_tcn = pd.concat([exp_data_tcn, data_tcn], axis=0, join='inner')
        if i ==0:
            # exp_data = e
            exp_data_tcn = e_tcn
            # exp_data_noalign = e_noalign
            # exp_data_student = e_student
            # exp_data_student_noalign = e_student_noalign
            exp_data_student_generalization = e_student_generalization
            exp_data_student_generalization_noalign = e_student_generalization_noalign
        else:
            # exp_data = pd.concat([exp_data, e], axis=0, join='inner')
            exp_data_tcn = pd.concat([exp_data_tcn, e_tcn], axis=0, join='inner')
            # exp_data_noalign = pd.concat([exp_data_noalign, e_noalign], axis=0, join='inner')
            # exp_data_student = pd.concat([exp_data_student, e_student], axis=0, join='inner')
            # exp_data_student_noalign = pd.concat([exp_data_student_noalign, e_student_noalign], axis=0, join='inner')
            exp_data_student_generalization = pd.concat([exp_data_student_generalization, e_student_generalization], axis=0, join='inner')
            exp_data_student_generalization_noalign = pd.concat([exp_data_student_generalization_noalign, e_student_generalization_noalign], axis=0, join='inner')
    x_column = "shape"
    y_column = "sum success"
    # y_column = "termination step"
    results = pd.concat([exp_data_student_generalization, exp_data_student_generalization_noalign, exp_data_tcn], axis=0, join='inner')
    plt.xlabel(x_column, fontsize=20)
    plt.ylabel("success rate [%]", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18) 
    # Plot the experimental results
    #sns.set(style="whitegrid")  # Set Seaborn style
    # ax=sns.barplot(x=x_column, y=y_column, hue="method", data=results,errorbar="sd", edgecolor="black",
    # err_kws={'linewidth': 3.0},
    # capsize = 0.05)
    sns.set_palette("colorblind")
    ax = sns.boxplot(x=x_column, y=y_column, hue="method", data=results, linewidth=3)
    # sns.stripplot(x=x_column, y=y_column, hue="method", data=results, dodge=True, palette='dark:black', ax=ax)
    sns.stripplot(x=x_column, y=y_column, hue="method", data=results, size=9, dodge=True, edgecolor="black",linewidth=1, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], fontsize=18)
    # plt.legend(fontsize=18)
    # plt.tight_layout()
    plt.savefig("results_generalization"+ "hole" +str(10)+ "angle" + str(angle)+"_success.png")
    plt.show()