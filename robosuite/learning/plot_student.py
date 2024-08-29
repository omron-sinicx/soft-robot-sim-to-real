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

def list_out(x):
    return ','.join(x)

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
    

    path = "./student_progress/02-24-analysis/02-05-circle-circle-peg-hole-pegangle-seed-10-5-3-student_states.csv"
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    #print(df.iloc[:,[3,4]])
    palette=sns.color_palette(['#CC6677','#0077BB', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499']) 
    # sns.set_palette(palette)
    # palette=sns.color_palette("colorblind") 
    sns.set_palette(palette)
    fig, axes = plt.subplots(2, 2)
    t = np.arange(0,0.05*len(df),0.05)
    dt = pd.DataFrame(t.T, columns =["26"])
    r = pd.concat([df,dt],axis=1)
    sns.lineplot(x=r.iloc[:,26],y=r.iloc[:,15], ax=axes[0,0], linestyle='--',linewidth = 4, ci=None)
    sns.lineplot(x=r.iloc[:,26],y=r.iloc[:,25], ax=axes[0,0],linewidth = 4)
    axes[0,0].set_xlabel("time [s]", fontsize= 20)
    axes[0,0].set_ylabel("alignment", fontsize= 20)
    axes[0,0].tick_params(axis='both', which='major', labelsize=18)
    axes[0,0].legend(["grand truth","predicted"],fontsize= 18)

    sns.lineplot(x=r.iloc[:,26],y=r.iloc[:,6],ax=axes[0,1], linestyle='--',linewidth = 4, ci=None)#,x=r.iloc[:,6], y=r.iloc[:,17])
    sns.lineplot(x=r.iloc[:,26],y=r.iloc[:,16],ax=axes[0,1], linewidth = 4, ci=None)
    axes[0,1].set_xlabel("time [s]", fontsize= 20)
    axes[0,1].set_ylabel("x position [m]", fontsize= 20)
    axes[0,1].tick_params(axis='both', which='major', labelsize=18)

    # sns.lineplot(data=r.iloc[:,[17,7]],x=r.iloc[:,26],y=r.iloc[:,17],ax=axes[1,0])#],x=r.iloc[:,16], y=r.iloc[:,17])
    sns.lineplot(x=r.iloc[:,26],y=r.iloc[:,7],ax=axes[1,0], linestyle='--',linewidth = 4, ci=None)#,x=r.iloc[:,6], y=r.iloc[:,17])
    sns.lineplot(x=r.iloc[:,26],y=r.iloc[:,17],ax=axes[1,0], linewidth = 4, ci=None)
    axes[1,0].set_xlabel("time [s]", fontsize= 20)
    axes[1,0].set_ylabel("y position [m]", fontsize= 20)
    axes[1,0].tick_params(axis='both', which='major', labelsize=18)
    
    # sns.lineplot(data=r.iloc[:,[18,8]],x=r.iloc[:,26],y=r.iloc[:,18], ax=axes[1,1])#],x=r.iloc[:,16], y=r.iloc[:,17])
    sns.lineplot(x=r.iloc[:,26],y=r.iloc[:,8],ax=axes[1,1], linestyle='--',linewidth = 4, ci=None)#,x=r.iloc[:,6], y=r.iloc[:,17])
    sns.lineplot(x=r.iloc[:,26],y=r.iloc[:,18],ax=axes[1,1], linewidth = 4, ci=None)
    axes[1,1].set_xlabel("time [s]", fontsize= 20)
    axes[1,1].set_ylabel("z position [m]", fontsize= 20)
    axes[1,1].tick_params(axis='both', which='major', labelsize=18)
    # plt.savefig("results"+ "hole" +str(10)+ "angle" + str(angle)+"_success.png")
    plt.show()