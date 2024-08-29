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

if __name__ == "__main__":
    # Specify the path to your CSV file
    date = "01-19"
    date_tcn = "02-07"
    obj = ["circle", "square", "triangle", "rectangle"]
    angle = 5
    #fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(1):
        exp_data = []
        exp_data_tcn = []
        for seed in range(10):
            name = date + "-" + obj[i] + "-peg-hole-pegangle-seed-10-" + str(angle) +"-" + str(seed)
            name_tcn = date_tcn + "-tcn-" + obj[i] + "-peg-hole-pegangle-10-" + str(angle) +"-" + str(seed)
            csv_file_path = "./learning_progress/" + name + "/" + name + "_stats.csv"
            csv_file_path_tcn = "./learning_progress_tcn/" + name_tcn + "/" + name_tcn + "_stats.csv"
            # Load data from the CSV file
            d = load_data(csv_file_path)
            c = pd.DataFrame({'method':["privileged"]*(len(d))})
            data = pd.concat([d,c], axis=1)
            dd = load_data(csv_file_path_tcn)
            cc = pd.DataFrame({'method':["TCN"]*(len(d))})
            data_tcn = pd.concat([dd,cc], axis=1)
            if seed == 0:
                exp_data = data
                exp_data_tcn = data_tcn.iloc[0:1000]
            else:
                exp_data = pd.concat([exp_data, data], axis=0, join='inner')
                exp_data_tcn = pd.concat([exp_data_tcn, data_tcn.iloc[0:1000]], axis=0, join='inner')
        #print(exp_data["mean reward"][:])
        results = pd.concat([exp_data, exp_data_tcn], axis=0, join='inner')
        # Specify the column names for the x and y axes
        x_column = "iteration"
        y_column = "mean reward"
        #['0173B2', 'DE8F05', '029E73', 'D55E00', 'CC78BC', 'CA9161', 'FBAFE4', '949494', 'ECE133', '56B4E9']
        palette=sns.color_palette(['#CC6677', '#029E73', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499']) 
        sns.set_palette(palette)
        # Create a scatter plot
        #plt.figure(figsize=(10, 6))  # Adjust figure size if needed
        # axes[i].set_title(obj[i])
        sns.lineplot(x=x_column, y=y_column, data=results, hue="method", errorbar='sd') 
        # sns.lineplot(x=x_column, y=y_column, data=exp_data_tcn, errorbar='sd')
        plt.xlabel(x_column, fontsize=20)
        plt.ylabel("reward", fontsize=20)
        plt.legend(fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=18)
        
        # Plot the experimental results
        # sns.set(style="whitegrid")  # Set Seaborn style

    # plt.tight_layout()
    plt.savefig("results"+ "hole" +str(10)+ "angle" + str(angle)+".png")
    plt.show()